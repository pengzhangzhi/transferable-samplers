"""
FastFlow: Fast Sampling, Exact Likelihood via Forward-Only Training

FastFlow trains a one-step normalizing flow using forward-only distillation:
- Teacher f: trained with MLE, forward is x→z (encoding, fast)
- Student g: trained with FastFlow, forward is z→x (generation, fast)

Training uses only forward computations: f(x), g(z), and f(g(z)).
No slow inverse operations are required during training.

Stage A (Teacher MLE - done separately):
    for x: z = f(x), L_f = 0.5||z||² - log|det J_f(x)|

Stage B (Student Forward-Only Training - this module):
    for x:
        z = f(x)                    (no grad, teacher encoding)
        x_hat = g(z_tilde)          (student decoding)
        L_rec = ||x - x_hat||²      (reconstruction)
        L_align = Σ||h_T - A(h_S)||² (feature alignment, early training)
        L_kl = KL(p_g || p_f)       (reverse KL, late training)
"""

import logging
from copy import deepcopy
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.neural_networks.ema import EMA
from src.models.transferable_boltzmann_generator_module import TransferableBoltzmannGeneratorLitModule

logger = logging.getLogger(__name__)


class FastFlowLitModule(TransferableBoltzmannGeneratorLitModule):
    """FastFlow: One-step sampling via forward-only distillation."""

    def __init__(
        self,
        net: torch.nn.Module,
        teacher_ckpt_path: str,
        adapter_dim: int = 256,
        lambda_rec: float = 1.0,
        lambda_align: float = 1.0,
        lambda_kl: float = 0.1,
        noise_sigma: float = 0.1,
        align_until_frac: float = 0.5,
        kl_start_frac: float = 0.67,
        *args,
        **kwargs,
    ) -> None:
        """Initialize FastFlowLitModule.

        Args:
            net: Student network (same architecture as teacher).
            teacher_ckpt_path: Path to pretrained teacher checkpoint.
            adapter_dim: Hidden dimension for feature alignment adapters.
            lambda_rec: Weight for reconstruction loss.
            lambda_align: Weight for feature alignment loss.
            lambda_kl: Weight for reverse KL loss.
            noise_sigma: Max noise std for latent jittering.
            align_until_frac: Fraction of training to use alignment loss.
            kl_start_frac: Fraction of training to start KL loss.
        """
        super().__init__(net=net, *args, **kwargs)

        # Save FastFlow-specific hyperparameters
        self.save_hyperparameters(
            ignore=["net", "datamodule"],
            logger=False,
        )

        # Create and load teacher (frozen)
        self.teacher = self._create_and_load_teacher(teacher_ckpt_path)

        # Create adapters for feature alignment
        self._create_adapters(adapter_dim)

    def _get_student_model(self) -> nn.Module:
        """Get the underlying student model (handles EMA wrapping)."""
        if isinstance(self.net, EMA):
            return self.net.model
        return self.net

    def _create_and_load_teacher(self, ckpt_path: str) -> nn.Module:
        """Create teacher from student architecture and load pretrained weights."""
        logger.info(f"Loading teacher from: {ckpt_path}")

        # Deepcopy the student architecture for teacher
        student_model = self._get_student_model()
        teacher = deepcopy(student_model)

        # Load weights
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Handle EMA checkpoint format (keys have 'net.model.' prefix)
        if any(k.startswith("net.model.") for k in state_dict.keys()):
            logger.info("Detected EMA checkpoint format, stripping prefixes...")
            new_state_dict = {}
            for k, v in state_dict.items():
                if "shadow_params" in k or "num_updates" in k:
                    continue
                if k.startswith("net.model."):
                    new_key = k[len("net.model."):]
                    new_state_dict[new_key] = v
                elif k.startswith("net."):
                    new_key = k[len("net."):]
                    new_state_dict[new_key] = v
            state_dict = new_state_dict

        teacher.load_state_dict(state_dict)

        # Freeze teacher
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        logger.info("Teacher loaded and frozen successfully")
        return teacher

    def _create_adapters(self, adapter_dim: int) -> None:
        """Create lightweight adapters for feature alignment."""
        student_model = self._get_student_model()
        num_blocks = len(student_model.blocks)

        self.adapters = nn.ModuleList([
            nn.Linear(adapter_dim, adapter_dim)
            for _ in range(num_blocks)
        ])

        logger.info(f"Created {num_blocks} adapters with dim={adapter_dim}")

    def model_step(self, batch: dict[str, torch.Tensor], log: bool = True) -> torch.Tensor:
        """FastFlow training step with reconstruction, alignment, and reverse KL.

        Loss scheduling:
        - Early training (epoch < align_until_frac): L_rec + L_align
        - Late training (epoch >= kl_start_frac): L_rec + L_kl
        """
        x = batch["x"]

        student = self._get_student_model()

        # Compute epoch fraction for loss scheduling
        if self.trainer is not None and self.trainer.max_epochs > 0:
            epoch_frac = self.current_epoch / self.trainer.max_epochs
        else:
            epoch_frac = 0.0

        use_align = epoch_frac < self.hparams.align_until_frac
        use_kl = epoch_frac >= self.hparams.kl_start_frac

        # ===== RECONSTRUCTION LOSS =====
        # Teacher encoding (no grad)
        with torch.no_grad():
            z_teacher, _, feats_t = self.teacher(x, return_intermediates=True)
            # Clamp for stability
            feats_t = [torch.clamp(h, -1, 1) for h in feats_t]

            z_input = z_teacher

        # Student decoding
        x_hat, _, feats_s = student(z_input, return_intermediates=True)

        # Reconstruction loss
        L_rec = F.mse_loss(x_hat, x)

        # ===== ALIGNMENT LOSS (early training) =====
        L_align = torch.zeros((), device=x.device)
        if use_align:
            for ft, fs, adapter in zip(feats_t, feats_s, self.adapters):
                L_align = L_align + F.mse_loss(adapter(fs), ft.detach())
            L_align = L_align / len(self.adapters)

        # ===== REVERSE KL LOSS (late training) =====
        L_kl = torch.zeros((), device=x.device)
        kl_metrics = {}
        if use_kl:
            L_kl, kl_metrics = self._compute_reverse_kl(x.shape[0], student)

        # Total loss
        loss = (
            self.hparams.lambda_rec * L_rec
            + self.hparams.lambda_align * L_align
            + self.hparams.lambda_kl * L_kl
        )

        # Logging (train/loss is logged by parent class via train_metrics)
        if log:
            self.log("train/L_rec", L_rec.item(), prog_bar=True, sync_dist=True)
            self.log("train/L_align", L_align.item(), prog_bar=True, sync_dist=True)
            self.log("train/L_kl", L_kl.item(), prog_bar=True, sync_dist=True)
            self.log("train/use_align", float(use_align), sync_dist=True)
            self.log("train/use_kl", float(use_kl), sync_dist=True)
            for k, v in kl_metrics.items():
                self.log(f"train/kl_{k}", v, sync_dist=True)

        return loss

    def _compute_reverse_kl(
        self, batch_size: int, student: nn.Module
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute reverse KL: KL(p_g || p_f) using forward-only computations.

        Formula: loss = z_norm - logdet_f - logdet_g + z_recon

        Where:
        - z_norm = 0.5 * ||f(g(z))||² (teacher's encoding of student samples)
        - z_recon = ||f(g(z)) - z||² (cycle consistency)
        - logdet_f = teacher's logdet
        - logdet_g = student's logdet
        """
        num_atoms = self.datamodule.hparams.num_atoms

        # Sample z from prior
        z = self.prior.sample(batch_size, num_atoms, device=self.device)

        # Student forward: z → x (fast, one pass)
        x_sample, logdet_g = student(z)

        # Teacher forward: x → z' (evaluates likelihood of student samples)
        z_teacher, logdet_f = self.teacher(x_sample)

        # Mean logdets over batch
        logdet_g = logdet_g.mean()
        logdet_f = logdet_f.mean()

        # z_norm: 0.5 * ||z'||² (should be ~0.5 if z' ~ N(0,1))
        z_norm = 0.5 * z_teacher.pow(2).mean()

        # Cycle consistency: ||z' - z||² (should be ~0 if g inverts f)
        z_recon = (z_teacher - z).pow(2).mean()

        # Reverse KL loss (up to constants)
        # log p_f(x) = -0.5||z'||² + logdet_f, so -log p_f(x) = z_norm - logdet_f
        # We also add cycle consistency for regularization
        loss = z_norm - logdet_f - logdet_g + z_recon

        # Debug: log statistics of z and z'
        metrics = {
            "z_norm": z_norm.item(),
            "z_recon": z_recon.item(),
            "logdet_f": logdet_f.item(),
            "logdet_g": logdet_g.item(),
            "z_std": z.std().item(),
            "z_teacher_std": z_teacher.std().item(),
            "x_sample_std": x_sample.std().item(),
        }

        return loss, metrics

    def generate_samples(
        self,
        batch_size: int,
        permutations: Optional[dict[str, torch.Tensor]] = None,
        encodings: Optional[dict[str, torch.Tensor]] = None,
        n_timesteps: int = None,
        dummy_ll: bool = False,
        log_invert_error: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fast one-step sampling using student's forward pass.

        Unlike standard NF which uses slow reverse(), FastFlow uses fast forward().

        Returns:
            x_pred: Generated samples
            log_q: Log-likelihood of samples
            prior_samples: The noise samples used for generation
        """
        if encodings is None:
            num_atoms = self.datamodule.hparams.num_atoms
        else:
            num_atoms = encodings["atom_type"].size(0)

        data_dim = num_atoms * self.datamodule.hparams.num_dimensions

        local_batch_size = batch_size // self.trainer.world_size
        prior_samples = self.prior.sample(local_batch_size, num_atoms, device=self.device)

        # Prior log probability
        prior_log_q = -self.prior.energy(prior_samples) * data_dim

        student = self._get_student_model()

        with torch.no_grad():
            # FAST: student.forward(z) → x directly (one parallel pass!)
            x_pred, fwd_logdets = student(prior_samples)
            fwd_logdets = fwd_logdets * data_dim

            # DDP all_gather
            x_pred = self.all_gather(x_pred).reshape(-1, *x_pred.shape[1:])
            fwd_logdets = self.all_gather(fwd_logdets).reshape(-1)
            prior_log_q = self.all_gather(prior_log_q).reshape(-1)
            prior_samples = self.all_gather(prior_samples).reshape(-1, *prior_samples.shape[1:])

        # NOTE: Minus sign! Student trained z→x, so:
        # log p_g(x) = log p(z) - log|det J_g(z)|
        log_q = prior_log_q - fwd_logdets

        return x_pred, log_q, prior_samples

    def proposal_energy(
        self,
        x: torch.Tensor,
        permutations: Optional[dict[str, torch.Tensor]] = None,
        encodings: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute proposal energy using TEACHER (for SMC compatibility).

        The teacher was trained x→z, so teacher.forward(x) gives meaningful likelihood.
        This is used for importance sampling and SMC refinement.
        """
        data_dim = x.shape[1] * self.datamodule.hparams.num_dimensions

        # Teacher encodes x→z
        # z, logdet = self.teacher(x)
        student = self._get_student_model()
        z, logdet = student.reverse(x, return_logdets=True)
        logdet = logdet * data_dim
        prior_energy = self.prior.energy(z) * data_dim

        # For teacher (x→z): log p(x) = log p(z) + logdet
        # So energy = -log p(x) = prior_energy - logdet
        energy = prior_energy - logdet

        if self.hparams.sampling_config.get("use_com_adjustment", False):
            com_energy = self.com_energy_adjustment(x)
            energy = energy - com_energy

        return energy

    def com_energy_adjustment(self, x: torch.Tensor) -> torch.Tensor:
        """Center of mass energy adjustment (same as NormalizingFlowLitModule)."""
        import math
        import scipy

        assert self.proposal_com_std is not None, "Center of mass std should be set"

        sigma = self.proposal_com_std
        com = x.mean(dim=1, keepdim=False)
        com_norm = com.norm(dim=-1)
        com_energy = com_norm**2 / (2 * sigma**2) - torch.log(
            com_norm**2 / (math.sqrt(2) * sigma**3 * scipy.special.gamma(3 / 2))
        )

        return com_energy


if __name__ == "__main__":
    _ = FastFlowLitModule(None, None, None, None)

