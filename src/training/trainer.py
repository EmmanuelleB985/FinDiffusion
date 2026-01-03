"""Training loop for FinDiffusion."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Optimization
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    clip_grad_norm: float = 1.0
    
    # Scheduler
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Mixed precision
    use_amp: bool = True
    
    # Logging
    log_every: int = 100
    sample_every: int = 10
    save_every: int = 10
    
    # Classifier-free guidance
    drop_cond_prob: float = 0.1
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    
    # Wandb
    use_wandb: bool = False
    project: str = "fin-diffusion"
    run_name: Optional[str] = None


class Trainer:
    """
    Trainer for FinDiffusion model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        
        # Device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if self.config.use_amp else None
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
        # Checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Wandb
        self.wandb_run = None
        if self.config.use_wandb:
            self._init_wandb()

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        warmup_steps = len(self.train_loader) * self.config.warmup_epochs
        total_steps = len(self.train_loader) * self.config.epochs
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.min_lr,
        )
        
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config.project,
                name=self.config.run_name,
                config={
                    "epochs": self.config.epochs,
                    "lr": self.config.lr,
                    "batch_size": self.train_loader.batch_size,
                }
            )
        except ImportError:
            logger.warning("wandb not installed, skipping wandb logging")
            self.config.use_wandb = False

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            if isinstance(batch, torch.Tensor):
                x = batch.to(self.device)
            else:
                x = batch["returns"].to(self.device)
            
            # Forward pass with optional AMP
            self.optimizer.zero_grad()
            
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(x, drop_cond_prob=self.config.drop_cond_prob)
                    loss = outputs["loss"]
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.clip_grad_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(x, drop_cond_prob=self.config.drop_cond_prob)
                loss = outputs["loss"]
                loss.backward()
                
                if self.config.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.clip_grad_norm
                    )
                
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix(loss=loss.item(), lr=self.scheduler.get_last_lr()[0])
            
            # Log to wandb
            if self.config.use_wandb and batch_idx % self.config.log_every == 0:
                import wandb
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "global_step": self.global_step,
                })
        
        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> float:
        """Validate the model."""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            if isinstance(batch, torch.Tensor):
                x = batch.to(self.device)
            else:
                x = batch["returns"].to(self.device)
            
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(x, drop_cond_prob=0.0)
                    loss = outputs["loss"]
            else:
                outputs = self.model(x, drop_cond_prob=0.0)
                loss = outputs["loss"]
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches

    @torch.no_grad()
    def generate_samples(
        self,
        n_samples: int = 16,
        conditions: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Generate sample sequences."""
        self.model.eval()
        return self.model.generate(
            n_samples=n_samples,
            conditions=conditions,
            device=self.device,
            progress=False,
        )

    def save_checkpoint(self, filename: str = "checkpoint.pt", is_best: bool = False):
        """Save a checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": self.config.__dict__,
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def load_checkpoint(self, path: Union[str, Path]):
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        
        logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")

    def train(self):
        """Full training loop."""
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Log
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )
            
            # Wandb logging
            if self.config.use_wandb:
                import wandb
                log_dict = {
                    "epoch": epoch + 1,
                    "train/epoch_loss": train_loss,
                    "val/epoch_loss": val_loss,
                }
                
                # Generate and log samples
                if (epoch + 1) % self.config.sample_every == 0:
                    samples = self.generate_samples(n_samples=8)
                    # Could add visualization here
                
                wandb.log(log_dict)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch{epoch + 1}.pt", is_best=is_best)
        
        # Final save
        self.save_checkpoint("final.pt")
        
        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time / 3600:.2f} hours")
        
        if self.config.use_wandb:
            import wandb
            wandb.finish()


class EMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Apply EMA weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
