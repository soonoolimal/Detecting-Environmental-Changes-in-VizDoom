import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm

from trainer.base_trainer import BaseTrainer


class AETrainer(BaseTrainer):
    """ObsAutoEncoder Trainer (Denoising MSE)

    Runs a warmup-style limited training (typically small epoch count + tight patience)
    intended as a pretraining step, not full training.
    """
    def __init__(
        self,
        model,
        device,
        denoise_std: float = 0.05,
        **kwargs,
    ):
        super().__init__(model, device, ckpt_name="best_ae.pt", **kwargs)
        
        self.denoise_std = denoise_std

    def build_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def train_step(self, batch) -> torch.Tensor:
        ob = batch.to(self.device)  # (B,C,H,W) float32 in [0,1]

        if self.denoise_std > 0:
            ob_in = (ob + self.denoise_std * torch.randn_like(ob)).clamp(0.0, 1.0)
        else:
            ob_in = ob

        ob_hat = self.model(ob_in)
        loss = F.mse_loss(ob_hat, ob)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip is not None:
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss

    def _eval(self, valid_loader, epoch: int) -> float:
        total_loss, steps = 0.0, 0
        with torch.no_grad():
            pbar = tqdm(valid_loader, total=len(valid_loader), leave=False, desc="[Eval AE]")
            for ob in pbar:
                ob = ob.to(self.device)
                ob_hat = self.model(ob)
                total_loss += float(F.mse_loss(ob_hat, ob))
                steps += 1
        val_loss = total_loss / max(steps, 1)
        
        return val_loss

    def pretrain(self, train_loader, valid_loader, epochs: int = 10):
        """Warmup Pretraining: Limited Epochs + Early Stopping
        
        Intended to initialize the backbone before DT training,
        not to fully converge the autoencoder.
        """
        self._run_loop(train_loader, valid_loader, epochs, desc="Pretrain AE")