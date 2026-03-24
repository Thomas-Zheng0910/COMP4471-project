from typing import Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from nystrom_attention import NystromAttention as _NystromAttention

    class NystromAttentionWrapper(nn.Module):
        """
        Thin wrapper around lucidrains' NystromAttention so it accepts
        pre-computed (q, k, v) tensors of shape (B, N, H, D) — the same
        convention used by AttentionBlock — instead of a single fused
        (B, N, dim) input tensor.

        lucidrains' NystromAttention expects a *single* tensor and does its
        own QKV projection internally, so we bypass that by monkey-patching
        the projection weights to identity-like behaviour and feeding the
        concatenated QKV directly.

        A simpler and fully transparent approach (used here) is to implement
        the Nyström steps manually using the landmark / kernel logic that
        lucidrains exposed, but the easiest zero-dependency path is to just
        run a plain scaled-dot-product softmax with the Nyström sketch.
        Because we already have q/k/v we implement the three-matrix Nyström
        approximation directly.
        """

        def __init__(self, num_landmarks: int, num_heads: int, dropout: float = 0.0):
            super().__init__()
            self.num_landmarks = num_landmarks
            self.num_heads = num_heads
            self.dropout = nn.Dropout(dropout)

        @staticmethod
        def _moore_penrose_iter_pinv(x: torch.Tensor, iters: int = 6) -> torch.Tensor:
            """Iterative Moore-Penrose pseudo-inverse (same trick lucidrains uses)."""
            abs_x = torch.abs(x)
            col = abs_x.sum(dim=-1, keepdim=True)
            row = abs_x.sum(dim=-2, keepdim=True)
            z = x.transpose(-1, -2) / (torch.max(col) * torch.max(row))
            I = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
            I = I.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
            for _ in range(iters):
                xz = x @ z
                z = 0.25 * z @ (13 * I - xz @ (15 * I - xz @ (7 * I - xz)))
            return z

        def forward(
            self,
            q: torch.Tensor,          # (B, N, H, D)
            k: torch.Tensor,          # (B, N, H, D)
            v: torch.Tensor,          # (B, N, H, D)
            key_padding_mask=None,    # (B, N) bool — True where *ignored*
        ) -> torch.Tensor:            # (B, N, H, D)
            B, N, H, D = q.shape
            L = self.num_landmarks
            scale = D ** -0.5

            # ---- landmark (segment-mean) queries & keys ----------------
            # Divide sequence into L buckets and average inside each bucket.
            # This is the standard Nyström landmark construction.
            # q/k: (B, N, H, D)  ->  reshape to (B, H, N, D) for matmuls
            q = q.permute(0, 2, 1, 3)  # (B, H, N, D)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            # Landmark construction via segment mean
            # Pad N to a multiple of L
            pad_len = (L - N % L) % L
            if pad_len:
                q_pad = F.pad(q, (0, 0, 0, pad_len))
                k_pad = F.pad(k, (0, 0, 0, pad_len))
            else:
                q_pad = q
                k_pad = k

            seg = q_pad.shape[2] // L  # tokens per landmark bucket
            q_landmarks = q_pad.reshape(B, H, L, seg, D).mean(dim=3)  # (B,H,L,D)
            k_landmarks = k_pad.reshape(B, H, L, seg, D).mean(dim=3)  # (B,H,L,D)

            # ---- three kernel matrices ---------------------------------
            # kernel_1: (B, H, N, L)
            kernel_1 = torch.softmax(q @ k_landmarks.transpose(-1, -2) * scale, dim=-1)
            # kernel_2: (B, H, L, L)  — pseudo-inverted below
            kernel_2 = torch.softmax(q_landmarks @ k_landmarks.transpose(-1, -2) * scale, dim=-1)
            # kernel_3: (B, H, L, N)
            kernel_3 = torch.softmax(q_landmarks @ k.transpose(-1, -2) * scale, dim=-1)

            # Apply key padding mask to kernel_3 (the part that touches keys)
            if key_padding_mask is not None:
                # key_padding_mask: (B, N), True = ignore
                mask = key_padding_mask[:, None, None, :]  # (B,1,1,N)
                kernel_3 = kernel_3.masked_fill(mask, 0.0)

            kernel_2_pinv = self._moore_penrose_iter_pinv(kernel_2)  # (B,H,L,L)

            # Nyström approximation: A ≈ K1 @ pinv(K2) @ K3
            # (B,H,N,L) @ (B,H,L,L) @ (B,H,L,N) @ (B,H,N,D)
            x = kernel_1 @ kernel_2_pinv @ (kernel_3 @ v)  # (B,H,N,D)
            x = self.dropout(x)
            x = x.permute(0, 2, 1, 3)  # (B, N, H, D)
            return x

except ImportError:
    print(
        "Cannot import nystrom_attention. "
        "Install it with:  pip install nystrom-attention"
    )

    class NystromAttentionWrapper(nn.Module):  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            raise NotImplementedError(
                "NystromAttention is not available. "
                "Please run:  pip install nystrom-attention"
            )


from .attention import AttentionBlock


class NystromBlock(AttentionBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: Optional[int] = None,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            cosine=cosine,
            gated=gated,
            layer_scale=layer_scale,
            context_dim=context_dim,
        )
        self.attention_fn = NystromAttentionWrapper(
            num_landmarks=128, num_heads=num_heads, dropout=dropout
        )

    def attn(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        pos_embed: Optional[torch.Tensor] = None,
        pos_embed_context: Optional[torch.Tensor] = None,
        rope: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        x = self.norm_attnx(x)
        context = self.norm_attnctx(context)
        k, v = rearrange(
            self.kv(context), "b n (kv h d) -> b n h d kv", h=self.num_heads, kv=2
        ).unbind(dim=-1)
        q = rearrange(self.q(x), "b n (h d) -> b n h d", h=self.num_heads)

        if rope is not None:
            q = rope(q)
            k = rope(k)
        else:
            if pos_embed is not None:
                pos_embed = rearrange(
                    pos_embed, "b n (h d) -> b n h d", h=self.num_heads
                )
                q = q + pos_embed
            if pos_embed_context is not None:
                pos_embed_context = rearrange(
                    pos_embed_context, "b n (h d) -> b n h d", h=self.num_heads
                )
                k = k + pos_embed_context

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim

        # NystromAttentionWrapper accepts (q, k, v) of shape (B, N, H, D)
        # and key_padding_mask of shape (B, N) — same contract as before.
        x = self.attention_fn(q, k, v, key_padding_mask=attn_bias)
        x = rearrange(x, "b n h d -> b n (h d)")
        x = self.out(x)
        return x