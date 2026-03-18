import os
from abc import ABC, abstractmethod
from typing import Optional
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer(ABC):
    """
    Common training loop with early stopping and best-checkpoint saving.
    """
    def __init__(
        self,
        model,
        device,
        ckpt_name,
        env_name: str,
        exp_name: str,
        timestamp: str,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        patience: int = 5,
        writer: Optional[SummaryWriter] = None,
    ):
        self.model = model
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.patience = patience
        self.writer = writer

        self.id = f"{env_name}_{exp_name}"
        self.save_dir = os.path.join(os.getcwd(), "model", "pretrained", f"{self.id}_{timestamp}")
        self.ckpt_name = ckpt_name

        model.to(device)
        self.optimizer = self.build_optimizer()

    @abstractmethod
    def build_optimizer(self) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def train_step(self, batch) -> torch.Tensor:
        """
        Runs forward and backward pass for single batch.
        Returns scalar training loss.
        """
        pass

    @abstractmethod
    def _eval(self, valid_loader, epoch: int) -> float:
        """
        Runs full validation step.
        Returns average validation loss.
        """
        pass

    def _log_train(self, epoch: int, train_loss: float):
        """Log training loss to TensorBoard. Override for extra metrics."""
        if self.writer is not None:
            tag = self.__class__.__name__.replace("Trainer", "")
            self.writer.add_scalar(f"{tag}/train_loss", train_loss, epoch)

    def _log_val(self, epoch: int, val_loss: float):
        """Log validation loss to TensorBoard. Override for extra metrics."""
        if self.writer is not None:
            tag = self.__class__.__name__.replace("Trainer", "")
            self.writer.add_scalar(f"{tag}/val_loss", val_loss, epoch)

    def _extra_ckpt(self) -> dict:
        """
        Extra state dicts to include in the checkpoint, e.g. GradScaler.
        Override to add entries.
        """
        return {}

    def _save_checkpoint(self, epoch: int, val_loss: float):
        os.makedirs(self.save_dir, exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            **self._extra_ckpt(),
        }
        path = os.path.join(self.save_dir, self.ckpt_name)
        torch.save(ckpt, path)
        
        return path

    def _run_loop(self, train_loader, valid_loader, epochs: int, desc: str):
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss, steps = 0.0, 0
            pbar = tqdm(train_loader, total=len(train_loader), leave=False)
            for batch in pbar:
                pbar.set_description(f"[{desc}] Epoch {epoch + 1}/{epochs}")
                
                loss = self.train_step(batch)
                total_loss += float(loss.detach())
                steps += 1
                
                pbar.set_postfix(dict(train_loss=round(total_loss / steps, 4)))

            train_loss = total_loss / max(steps, 1)
            print(f"[Epoch {epoch + 1}/{epochs}] Training Loss: {train_loss:.4f}")
            self._log_train(epoch + 1, train_loss)

            # Evaluation
            self.model.eval()
            val_loss = self._eval(valid_loader, epoch + 1)
            print(f"[Epoch {epoch + 1}/{epochs}] Validation Loss: {val_loss:.4f}")
            self._log_val(epoch + 1, val_loss)

            # Early Stopping + Save Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                path = self._save_checkpoint(epoch + 1, best_val_loss)
                print(f"\t Best checkpoint saved (validation loss {best_val_loss:.4f}): {path}")
            else:
                epochs_no_improve += 1
                print(f"\t No improvement ({epochs_no_improve}/{self.patience})")
                if epochs_no_improve >= self.patience:
                    print("Early stopping triggered.")
                    break

    def train(self, train_loader, valid_loader, epochs: int):
        self._run_loop(train_loader, valid_loader, epochs, desc=f"{self.id}, {self.__class__.__name__}")