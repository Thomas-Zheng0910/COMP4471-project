# Baseline NYUv2 Benchmark Reference

> Numbers from original papers — verify before citing.

## UniDepth V2 (Piccinelli et al., 2025)

- Depth type: **Metric** (no alignment needed)
- Evaluation: **Zero-shot** on NYUv2 (not in training set)

| Backbone | AbsRel | RMSE  | d1    |
|----------|--------|-------|-------|
| ViT-L    | 0.054  | 0.226 | 0.968 |

## Depth Anything V2 (Yang et al., NeurIPS 2024)

- Backbone: DINOv2 ViT + DPT decoder
- Eigen split, 654 test images, max depth 10m

**Metric depth (fine-tuned on NYUv2 indoor):**

| Variant | AbsRel | RMSE  | d1    |
|---------|--------|-------|-------|
| ViT-S   | 0.063  | 0.261 | 0.953 |
| ViT-B   | 0.056  | 0.236 | 0.963 |
| ViT-L   | 0.047  | 0.210 | 0.971 |

**Relative depth (scale-shift alignment at eval):**

| Variant | AbsRel | RMSE  | d1    |
|---------|--------|-------|-------|
| ViT-S   | 0.070  | 0.276 | 0.947 |
| ViT-B   | 0.058  | 0.238 | 0.962 |
| ViT-L   | 0.043  | 0.190 | 0.982 |

## Marigold (Ke et al., CVPR 2024)

- Depth type: **Relative** (affine-invariant, needs SSI alignment)
- Training: Synthetic only (Hypersim + Virtual KITTI 2), **zero-shot** on NYUv2
- Best results with 10 ensemble passes

| AbsRel | d1    | RMSE  | log10 |
|--------|-------|-------|-------|
| 0.055  | 0.964 | ~0.230| ~0.023|

## Notes

- **Metric vs Relative**: UniDepthV2 and DA-V2-Metric produce metric depth directly. Marigold and DA-V2-Relative need scale/shift alignment (SSI/SI) at eval time.
- **Fair comparison**: DA-V2-Metric is in-domain (fine-tuned on NYUv2); UniDepthV2 and Marigold are zero-shot.
- Our `eval_depth` computes both raw and SSI/SI-aligned metrics (`d1_ssi`, `arel_ssi`, etc.), so all baselines can be compared fairly.
