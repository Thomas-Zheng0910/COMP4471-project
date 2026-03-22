# LiDAR + Transformer 深度估計：目前進度與下一步計畫

日期：2026-03-22

## 已完成（Current Steps Done）

### 1) NYUv2 dataset 已支援 LiDAR 載入

已修改 `data/nyuv2_dataset.py`，新增可選 LiDAR 載入能力，並保持向後相容：

- 新增參數：
  - `use_lidar`
  - `lidar_root`
  - `lidar_depth_scale`
  - `lidar_h5_key`
  - `lidar_confidence_h5_key`
- 支援兩種 LiDAR 來源：
  1. `.mat` 檔內的 HDF5 key（例如 `lidar_depths`）
  2. 外部檔案資料夾（`npy/npz`）
- 每個 sample 新增欄位（有 LiDAR 時）：
  - `lidar_depth`: 稀疏深度
  - `lidar_mask`: 有效像素遮罩
  - `lidar_confidence`: 信心值（若無提供則退化為 mask float）
  - `has_lidar`: 是否有 LiDAR
- 已包含 flip augmentation 一致性（影像/深度/LiDAR 一起翻轉）。

### 2) 訓練入口已接上 LiDAR dataset 參數

已修改 `train/train_depth.py`：

- 新增 CLI 參數：
  - `--use_lidar`
  - `--lidar_root`
  - `--lidar_depth_scale`
  - `--lidar_h5_key`
  - `--lidar_confidence_h5_key`
- `ImageDataset(...)` 初始化已傳入上述 LiDAR 參數。
- 目前模型前向仍以原本 RGB-depth 流程為主（尚未做 Transformer 層級 fusion）。

### 3) Phase 0-1 可執行工具已建立

新增 `data/get_datasets/phase01_prepare_nyuv2_lidar.py`，可完成：

- Phase 0（定義/檢查）：
  - 驗證 NYUv2 基礎 key（images/depths）
  - 驗證 LiDAR 來源是否可讀
- Phase 1（資料工程產物）：
  - 產生 Eigen 對齊 train/test split 檔
  - 計算 LiDAR sparsity 與深度分佈統計
  - 產生 `phase01_summary.json`（供實驗追蹤）

### 4) Phase 2 已完成：LiDAR sparse supervision 已接入訓練

已修改 `train/train_depth.py`，在不改 backbone/decoder 架構下完成最小可行 LiDAR 監督：

- 新增訓練參數：
  - `--lidar_loss_weight`
- 新增 loss 組件：
  - `LiDARSparse`（`mean(|log(pred)-log(lidar)|)`，僅在 `lidar_mask` 有效像素上計算）
  - 可用 `lidar_confidence` 做加權
- 訓練與驗證皆已整合 LiDAR loss：
  - 總 loss：`L_total = L_original + λ_lidar * L_lidar_sparse`
- 已新增 TensorBoard 指標：
  - `train/lidar_loss_raw`
  - `train/lidar_valid_ratio`
  - `train/lidar_valid_pixels`
  - `epoch/train_lidar_valid_ratio`
  - `epoch/val_lidar_valid_ratio`

### 5) Phase 3 已完成：Transformer late fusion 已接入模型

已修改 `model/unidepthv1/decoder.py` 與 `model/unidepthv1/unidepthv1.py`：

- 新增可開關的 late fusion：
  - `use_lidar_fusion`（decoder config）
- 融合方式：
  - 將 `lidar_depth + lidar_mask + lidar_confidence` 編碼成 LiDAR token
  - 在 1/16 尺度透過 attention prompt 融合到 depth latents
  - 使用 learned gate 控制融合強度
- 模型輸出新增融合監控資訊：
  - `fusion_stats.lidar_used`
  - `fusion_stats.lidar_valid_ratio`
  - `fusion_stats.lidar_gate_mean`

### 6) Phase 4 已完成：驗證與 fallback 監控已接入

已修改 `train/train_depth.py`：

- 加入 LiDAR dropout（訓練時隨機 drop LiDAR）：
  - `--lidar_dropout_prob`
- 加入 fallback 驗證（val 同時計算 RGB-only 與 RGB+LiDAR）：
  - `--phase4_eval_fallback`
  - 監控 RMSE gap（RGB-only − RGB+LiDAR）
- 新增融合與 fallback 相關 TensorBoard 指標：
  - `train/fusion_lidar_used`
  - `train/fusion_lidar_valid_ratio`
  - `train/fusion_lidar_gate_mean`
  - `epoch/train_fusion_lidar_gate_mean`
  - `epoch/train_lidar_dropout_ratio`
  - `epoch/val_fusion_lidar_gate_mean`
  - `epoch/val_rmse_with_lidar`
  - `epoch/val_rmse_rgb_only_fallback`
  - `epoch/val_rmse_gap_rgb_only_minus_lidar`

---

## 目前限制（What is NOT done yet）

1. 尚未完成 token fusion 消融：
  - 目前已完成 late fusion，尚未完成 token-level fusion 分支。
2. 尚未完成完整消融矩陣：
  - 尚未系統性比較 RGB-only、LiDAR supervision-only、LiDAR fusion。
3. 尚未做標定誤差敏感度、跨場景泛化、缺失 LiDAR 退化測試。

---

## 下一步（Recommended Next Plan）

### 下一步 A：token fusion 消融

1. 加入 token-level fusion 分支。
2. 對比 late fusion vs token fusion 在相同訓練 budget 下的結果。
3. 比較記憶體/速度/精度 tradeoff，選最終方案。

### 下一步 B：完整 Phase 5 驗證

1. 消融矩陣：
   - RGB-only
   - RGB + LiDAR supervision only
   - RGB + LiDAR fusion
2. 場景/距離分桶評估，觀察遠距與邊界區域。
3. LiDAR 缺失 fallback 測試（強制關 LiDAR）。
4. 標定擾動敏感度測試（intrinsics/extrinsics 小擾動）。
5. 跨場景泛化測試（房間/照明/材質分組）。

---

## 風險與排雷（Potential Problems）

1. **資料對齊錯誤**：外參或投影錯會直接污染 supervision。
2. **稀疏性過高**：有效像素比例過低時，訓練不穩。
3. **模型捷徑依賴 LiDAR**：若不做 dropout/mix，RGB-only 表現會崩。
4. **記憶體壓力**：Transformer 融合可能推高 VRAM，需漸進式導入。
5. **評估假象**：整體指標提升但關鍵場景退步，需要分桶分析。

---

## 建議立即執行指令

```bash
python -m data.get_datasets.phase01_prepare_nyuv2_lidar \
  --mat-path datasets/nyu_depth_v2_labeled.mat \
  --lidar-h5-key lidar_depths \
  --output-dir datasets/nyuv2_lidar_phase01
```

接著用 Phase 2 訓練（範例）：

```bash
python -m train.train_depth \
  --train_root datasets/nyu_depth_v2_labeled.mat \
  --val_root datasets/nyu_depth_v2_labeled.mat \
  --use_lidar true \
  --lidar_h5_key lidar_depths \
  --lidar_loss_weight 0.5 \
  --use_lidar_fusion true \
  --lidar_dropout_prob 0.2 \
  --phase4_eval_fallback true
```

Phase 3 checkpoint 推論（需同架構）：

```bash
python -m infer.infer_depth \
  --checkpoint runs/train_depth_xxx/checkpoints/epoch_100.pth \
  --data_root datasets/nyu_depth_v2_labeled.mat \
  --split test \
  --use_lidar_fusion true
```

或（若 LiDAR 在外部檔案）：

```bash
python -m data.get_datasets.phase01_prepare_nyuv2_lidar \
  --mat-path datasets/nyu_depth_v2_labeled.mat \
  --lidar-root datasets/nyuv2_lidar_projected \
  --output-dir datasets/nyuv2_lidar_phase01
```
