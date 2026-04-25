"""
Scheduled Self-Distillation for depth estimation.

Components
----------
EMATeacher     : EMA-updated copy of the model that runs the *hint* branch.
distill_lambda : Cosine warmup schedule → scalar weight in [0, 1].
confidence_mask: Entropy-based mask of teacher-confident spatial regions.
DistillationLoss: Multi-level loss combining soft-KL and feature-MSE.
"""

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── EMA Teacher ─────────────────────────────────────────────────────────────

class EMATeacher(nn.Module):
    """EMA wrapper around the hint branch parameters.

    The teacher's weights are never updated by gradient descent — only by an
    exponential moving average of the student model.  It is always run in eval
    mode.

    Args:
        student:        The student model (used only for parameter shapes when
                        no explicit teacher_model is given).
        alpha:          EMA decay; higher → teacher moves more slowly (0.999).
        teacher_model:  An already-built model to use as the teacher.  When
                        provided the student model is only used for EMA updates
                        (its parameters are mixed into the teacher via update()).
                        When None, the teacher is deep-copied from student.
    """

    def __init__(self, student: nn.Module, alpha: float = 0.999,
                 teacher_model: nn.Module = None):
        super().__init__()
        self.alpha = alpha
        self.model = teacher_model if teacher_model is not None else deepcopy(student)
        # Teacher must never receive gradients
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module):
        """EMA update for shared parameters only.

        Teacher and student may have different architectures (teacher has lidar
        fusion modules, student does not).  We match by parameter *name* and
        only update weights that exist in both models.  Teacher-only parameters
        (lidar fusion layers) are intentionally left unchanged — they were set
        once from the pre-trained teacher checkpoint and should not drift.
        """
        student_params = dict(student.named_parameters())
        for name, ema_p in self.model.named_parameters():
            if name in student_params:
                s_p = student_params[name]
                ema_p.data.mul_(self.alpha).add_(s_p.data, alpha=1 - self.alpha)
            # else: teacher-only param (e.g. lidar fusion) — keep as-is

    def forward(self, inputs: dict, image_metas: list):
        """Run teacher forward with hint inputs; returns (outputs, losses).

        Teacher is kept in eval mode so compute_losses returns empty losses
        (no wasted computation).  No gradient is tracked.
        """
        self.model.eval()
        with torch.no_grad():
            return self.model.forward_train(inputs, image_metas, force_compute_losses=False)


# ─── Distillation Schedule ───────────────────────────────────────────────────

def distill_lambda(t: int, T_warmup: int, T_total: int) -> float:
    """Cosine warmup schedule for the distillation loss weight.

    Returns 0 while t < T_warmup, then ramps from 0 → 1 via cosine curve.

    Args:
        t:        Current global training step.
        T_warmup: Number of warm-up steps before distillation starts.
        T_total:  Total training steps (ramp completes at T_total).

    Returns:
        Scalar weight in [0, 1].
    """
    if T_total <= T_warmup:
        return 1.0
    if t < T_warmup:
        return 0.0
    T_ramp = T_total - T_warmup
    x = min(t - T_warmup, T_ramp)
    return 0.5 * (1.0 - math.cos(math.pi * x / T_ramp))


# ─── Confidence Masking ──────────────────────────────────────────────────────

def confidence_mask(
    teacher_logits: torch.Tensor,
    entropy_threshold: float = 0.5,
) -> torch.Tensor:
    """Spatial mask selecting regions where the teacher is confident.

    Confidence is measured as *low* entropy of the channel-wise softmax
    distribution over the teacher's feature logits.

    Args:
        teacher_logits:    (B, C, H, W) multi-channel feature tensor.
        entropy_threshold: Pixels with entropy < threshold are kept.

    Returns:
        Boolean mask of shape (B, H, W); True where teacher is confident.
    """
    probs = F.softmax(teacher_logits, dim=1)                    # (B, C, H, W)
    log_probs = torch.clamp(probs, min=1e-10).log()             # (B, C, H, W)
    entropy = -(probs * log_probs).sum(dim=1)                   # (B, H, W)
    return entropy < entropy_threshold                           # bool (B, H, W)


# ─── Distillation Loss ───────────────────────────────────────────────────────

class DistillationLoss(nn.Module):
    """Multi-level distillation loss.

    Combines:
      - Output-level soft KL divergence (temperature-scaled) on cond_features.
      - Feature-level MSE alignment on cond_features (or any encoder feature).

    Teacher inputs must already be detached (``EMATeacher`` enforces this).

    Args:
        temperature:       Temperature for soft-KL (default 4.0).
        lambda_logit:      Weight for the KL component (default 1.0).
        lambda_feat:       Weight for the MSE component (default 0.1).
        entropy_threshold: Threshold for confidence masking (default 0.5).
    """

    def __init__(
        self,
        temperature: float = 4.0,
        lambda_logit: float = 1.0,
        lambda_feat: float = 0.1,
        entropy_threshold: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_logit = lambda_logit
        self.lambda_feat = lambda_feat
        self.entropy_threshold = entropy_threshold

    # ------------------------------------------------------------------
    def soft_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """KL(student || teacher) with temperature scaling.

        Teacher gradient is blocked inside this method as a safety net;
        callers should already pass detached teacher tensors.

        Args:
            student_logits: (B, C, H, W)
            teacher_logits: (B, C, H, W)  — teacher values, no grad required
            mask:           (B, H, W) bool; if None, all spatial positions used

        Returns:
            Scalar loss.
        """
        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=1)       # (B,C,H,W)
        teacher_probs = F.softmax(teacher_logits.detach() / T, dim=1)      # (B,C,H,W)

        # F.kl_div(input=log_Q, target=P) computes P*(log P - log Q) element-wise
        kl = F.kl_div(
            student_log_probs, teacher_probs, reduction="none"
        ).sum(dim=1)    # (B, H, W)

        if mask is not None:
            mask_f = mask.to(dtype=kl.dtype)
            denom = mask_f.sum().clamp(min=1.0)
            return (kl * mask_f).sum() / denom
        return kl.mean()

    # ------------------------------------------------------------------
    def feature_align_loss(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """MSE between student and teacher intermediate features.

        Args:
            student_feat: (B, C, H, W)
            teacher_feat: (B, C, H, W)  — teacher values, no grad required
            mask:         (B, H, W) bool; if None, all spatial positions used

        Returns:
            Scalar loss.
        """
        diff_sq = (student_feat - teacher_feat.detach()) ** 2   # (B, C, H, W)

        if mask is not None:
            mask_4d = mask.unsqueeze(1).to(dtype=diff_sq.dtype)  # (B,1,H,W)
            denom = (mask_4d.sum() * diff_sq.shape[1]).clamp(min=1.0)
            return (diff_sq * mask_4d).sum() / denom
        return diff_sq.mean()

    # ------------------------------------------------------------------
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_feat: torch.Tensor = None,
        teacher_feat: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute combined distillation loss with confidence masking.

        Args:
            student_logits: (B, C, H, W) output-level features, no-hint branch.
            teacher_logits: (B, C, H, W) output-level features, hint branch.
            student_feat:   (B, C', H', W') intermediate features, no-hint branch.
            teacher_feat:   (B, C', H', W') intermediate features, hint branch.

        Returns:
            Scalar combined loss:
                lambda_logit * soft_kl + lambda_feat * feat_mse
        """
        # Build confidence mask at logit resolution
        logit_mask = confidence_mask(
            teacher_logits.detach(), self.entropy_threshold
        )                                                       # (B, H, W)

        kl = self.soft_kl_loss(student_logits, teacher_logits, logit_mask)

        feat_loss = torch.tensor(0.0, device=student_logits.device)
        if student_feat is not None and teacher_feat is not None:
            # Resize mask to feature spatial resolution if needed
            feat_mask = logit_mask
            if student_feat.shape[-2:] != logit_mask.shape[-2:]:
                feat_mask = F.interpolate(
                    logit_mask.unsqueeze(1).float(),
                    size=student_feat.shape[-2:],
                    mode="nearest",
                ).squeeze(1).bool()
            feat_loss = self.feature_align_loss(student_feat, teacher_feat, feat_mask)

        return self.lambda_logit * kl + self.lambda_feat * feat_loss
