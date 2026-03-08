"""
DINOv2 ViT backbone stub.

To enable ViT-based training, implement ``_make_dinov2_model`` so that it
returns a ``DinoVisionTransformer`` instance with the following contract:

    model.embed_dims : List[int]
        Length = output_idx[-1].  For ViT-L/14 (depth=24, embed_dim=1024)
        this is ``[1024] * 24``.  Used by the decoder's ``ListAdapter`` to
        determine projection dimensions.

    model.depths : List[int]
        Equal to ``output_idx``, e.g. ``[5, 12, 18, 24]`` for ViT-L.
        Tells the decoder how to group and max-pool the encoder outputs.

    model.embed_dim : int
        Embedding dimension, e.g. 1024 for ViT-L.

    model(x) → (outputs, cls_tokens)
        outputs    : List[Tensor]  length = output_idx[-1], each [B, H/P, W/P, C]
        cls_tokens : List[Tensor]  length = output_idx[-1], each [B, 1, C]
        where P = patch_size (14 for DINOv2).

Reference implementation: the official UniDepth repository at
  https://github.com/lpiccinelli-eth/UniDepth/blob/main/unidepth/models/backbones/dinov2.py

Recommended output_idx values:
  ViT-L/14 (depth=24): [5, 12, 18, 24]
  ViT-B/14 (depth=12): [3,  6,  9, 12]
  ViT-S/14 (depth=12): [3,  6,  9, 12]

Note on image resolution for ViT:
  Prefer multiples of the patch size (14): 252, 336, 378, 392, 518 …
  The model's positional embeddings are interpolated to any size via
  DinoVisionTransformer.interpolate_pos_encoding(), but training at a
  patch-aligned resolution avoids rounding artefacts.
"""

# TODO:

def _make_dinov2_model(*args, **kwargs):
    raise NotImplementedError(
        "DINOv2 backbone is not implemented yet."
        "Use ConvNeXtV2 as the default encoder."
    )
