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

---

## 目前限制（What is NOT done yet）

1. 尚未完成「LiDAR + Transformer 特徵融合」：
   - 目前只完成資料載入與訓練入口參數對接。
2. 尚未在 loss 中使用 `lidar_depth/lidar_mask/lidar_confidence`：
   - 現行 loss 仍主要由原有 `depth/depth_mask` 驅動。
3. 尚未做標定誤差敏感度、跨場景泛化、缺失 LiDAR 退化測試。

---

## 下一步（Recommended Next Plan）

### Phase 2：最小可行 LiDAR 監督（先不改模型骨幹）

1. 在 training loop 將 `lidar_depth/lidar_mask/lidar_confidence` 取出。
2. 新增 sparse LiDAR loss（masked regression / SILog 變體）。
3. 總 loss 先採：
   - `L_total = L_rgb_depth + λ_lidar * L_lidar_sparse`
4. 增加 TensorBoard 指標：
   - LiDAR valid ratio
   - LiDAR-supervised pixel loss
   - LiDAR coverage bucket metrics

### Phase 3：Transformer 融合

1. 先做 late fusion（低風險）：
   - LiDAR sparse depth 經輕量 encoder
   - 在 decoder 中低解析度層插入 cross-attention
2. 再做 token fusion 消融：
   - 比較 late fusion vs token fusion
3. 加入 LiDAR dropout：
   - 避免模型過度依賴 LiDAR，保留 RGB-only 能力

### Phase 4：驗證與上線前檢查

1. 消融矩陣：
   - RGB-only
   - RGB + LiDAR supervision only
   - RGB + LiDAR fusion
2. 場景/距離分桶評估，觀察遠距與邊界區域。
3. LiDAR 缺失 fallback 測試（強制關 LiDAR）。

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

或（若 LiDAR 在外部檔案）：

```bash
python -m data.get_datasets.phase01_prepare_nyuv2_lidar \
  --mat-path datasets/nyu_depth_v2_labeled.mat \
  --lidar-root datasets/nyuv2_lidar_projected \
  --output-dir datasets/nyuv2_lidar_phase01
```
