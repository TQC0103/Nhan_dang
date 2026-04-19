# Colab End-to-End: SCRFD 2.5G Paper SR12

Mục tiêu của guide này:

- setup lại `SCRFD` trên Colab từ đầu
- train lại `SCRFD 2.5G` với đúng **12-scale SR set** của paper
- so sánh `baseline` và `ASR+JSAR`

Scale set paper `2.5GF`:

```python
[0.5, 0.7, 0.8, 1.0, 1.1, 1.2, 1.4, 1.5, 1.8, 2.0, 2.3, 2.6]
```

Config đã tạo sẵn:

- `configs/scrfd/scrfd_2.5g_80e_baseline_paper_sr12.py`
- `configs/scrfd/scrfd_2.5g_80e_asr_jsar_paper_sr12.py`
- `configs/scrfd/scrfd_2.5g_80e_cosine_baseline_paper_sr12.py`
- `configs/scrfd/scrfd_2.5g_80e_cosine_asr_jsar_paper_sr12.py`

## 1. Chuẩn bị Colab

Khuyên dùng:

- `GPU runtime`
- nếu có thể, chọn runtime có `2 GPU`

Nếu muốn lưu checkpoint ra Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2. Clone repo

Nếu dùng fork riêng:

```bash
%cd /content
!git clone <YOUR_FORK_URL> insightface
```

Nếu dùng repo gốc:

```bash
%cd /content
!git clone https://github.com/deepinsight/insightface.git
```

## 3. Setup env + dataset

```bash
%cd /content/insightface
!bash detection/scrfd/colab/setup_colab_env.sh /content/insightface
```

Script này sẽ:

- tạo env `micromamba`
- cài `torch`, `mmcv-full`, `mmdet`
- tải WIDERFace + SCRFD annotation
- chuẩn bị `detection/scrfd/data/retinaface`

Nếu bạn đã có dataset rồi và không muốn tải lại:

```bash
%cd /content/insightface
!SCRFD_PREPARE_RETINAFACE=0 SCRFD_DOWNLOAD_RETINAFACE=0 \
  bash detection/scrfd/colab/setup_colab_env.sh /content/insightface
```

## 4. Kiểm tra env

```bash
%cd /content/insightface
!bash detection/scrfd/colab/run_in_env.sh python - <<'PY'
import torch, mmcv, mmdet
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available(), 'count=', torch.cuda.device_count())
print('mmcv', mmcv.__version__)
print('mmdet', mmdet.__version__)
PY
```

## 5. Chạy baseline vs ASR+JSAR trên đúng paper SR12

### Cách nhanh nhất

Script wrapper đã trỏ sẵn sang config `paper_sr12`:

```bash
%cd /content/insightface/detection/scrfd
!bash colab/run_compare_2.5g_paper_sr12_baseline_vs_asr_jsar.sh
```

Mặc định script sẽ:

- train `baseline`
- train `ASR+JSAR`
- eval cả hai
- tạo bảng so sánh

Output mặc định:

- `work_dirs/compare_2.5g_paper_sr12/`
- `results/compare_2.5g_paper_sr12/`

### Nếu máy có 2 GPU

```bash
%cd /content/insightface/detection/scrfd
!BASELINE_GPU=0 IMPROVED_GPU=1 \
  bash colab/run_compare_2.5g_paper_sr12_baseline_vs_asr_jsar.sh
```

### Nếu muốn chỉnh batch size / workers / lr

Ví dụ:

```bash
%cd /content/insightface/detection/scrfd
!BASELINE_GPU=0 IMPROVED_GPU=1 \
  BATCH_SIZE_PER_GPU=32 WORKERS_PER_GPU=8 \
  BASELINE_LR=0.01 IMPROVED_LR=0.01 \
  bash colab/run_compare_2.5g_paper_sr12_baseline_vs_asr_jsar.sh
```

Lưu ý:

- script này **không tự scale lr theo batch**
- nếu tăng batch nhiều, chỉnh `BASELINE_LR` và `IMPROVED_LR` tay

## 6. Nếu muốn dùng cosine scheduler

Train tay:

```bash
%cd /content/insightface/detection/scrfd
!bash colab/run_in_env.sh python tools/train.py \
  configs/scrfd/scrfd_2.5g_80e_cosine_baseline_paper_sr12.py \
  --work-dir work_dirs/compare_2.5g_paper_sr12_cosine/baseline

!bash colab/run_in_env.sh python tools/train.py \
  configs/scrfd/scrfd_2.5g_80e_cosine_asr_jsar_paper_sr12.py \
  --work-dir work_dirs/compare_2.5g_paper_sr12_cosine/asr_jsar
```

Hoặc tự chạy song song bằng `CUDA_VISIBLE_DEVICES` nếu runtime có 2 GPU.

## 7. Eval lại với predictions để phân tích hard subset

Nếu bạn muốn giải thích vì sao `hard_AP` tăng, nên lưu `predictions/`:

```bash
%cd /content/insightface/detection/scrfd
!bash colab/run_in_env.sh python tools/test_widerface_enhanced.py \
  configs/scrfd/scrfd_2.5g_80e_baseline_paper_sr12.py \
  work_dirs/compare_2.5g_paper_sr12/baseline/latest.pth \
  --out results/compare_2.5g_paper_sr12/baseline \
  --save-preds

!bash colab/run_in_env.sh python tools/test_widerface_enhanced.py \
  configs/scrfd/scrfd_2.5g_80e_asr_jsar_paper_sr12.py \
  work_dirs/compare_2.5g_paper_sr12/asr_jsar/latest.pth \
  --out results/compare_2.5g_paper_sr12/asr_jsar \
  --save-preds
```

## 8. So sánh kết quả

```bash
%cd /content/insightface/detection/scrfd
!bash colab/run_in_env.sh python tools/compare_widerface_results.py \
  --baseline results/compare_2.5g_paper_sr12/baseline \
  --improved results/compare_2.5g_paper_sr12/asr_jsar \
  --baseline-name "SCRFD-2.5G PaperSR12 Baseline 80e" \
  --improved-name "SCRFD-2.5G PaperSR12 ASR+JSAR 80e" \
  --out-dir results/compare_2.5g_paper_sr12/comparison
```

File chính:

- `results/compare_2.5g_paper_sr12/comparison/comparison.md`

## 9. Vẽ scale probability history

```bash
%cd /content/insightface/detection/scrfd
!bash colab/run_in_env.sh python tools/plot_scale_prob_history.py \
  --source work_dirs/compare_2.5g_paper_sr12/asr_jsar \
  --config configs/scrfd/scrfd_2.5g_80e_asr_jsar_paper_sr12.py \
  --out-dir results/compare_2.5g_paper_sr12/scale_prob_history
```

## 10. Phân tích face-size distribution

```bash
%cd /content/insightface/detection/scrfd
!bash colab/run_in_env.sh python tools/analyze_sr_face_size_distribution.py \
  --ann-file data/retinaface/train/labelv2.txt \
  --baseline-config configs/scrfd/scrfd_2.5g_80e_baseline_paper_sr12.py \
  --improved-config configs/scrfd/scrfd_2.5g_80e_asr_jsar_paper_sr12.py \
  --improved-state work_dirs/compare_2.5g_paper_sr12/asr_jsar \
  --out-dir results/compare_2.5g_paper_sr12/face_size_analysis
```

## 11. Phân tích hard subset per-image

```bash
%cd /content/insightface/detection/scrfd
!bash colab/run_in_env.sh python tools/analyze_hard_subset_comparison.py \
  --baseline results/compare_2.5g_paper_sr12/baseline \
  --improved results/compare_2.5g_paper_sr12/asr_jsar \
  --gt-dir data/retinaface/val/gt \
  --out-dir results/compare_2.5g_paper_sr12/hard_subset_rerun
```

## 12. Gói artifact để tải về

```bash
%cd /content/insightface/detection/scrfd
!bash colab/run_in_env.sh python tools/package_analysis_artifacts.py \
  --experiment baseline work_dirs/compare_2.5g_paper_sr12/baseline results/compare_2.5g_paper_sr12/baseline configs/scrfd/scrfd_2.5g_80e_baseline_paper_sr12.py \
  --experiment asr_jsar work_dirs/compare_2.5g_paper_sr12/asr_jsar results/compare_2.5g_paper_sr12/asr_jsar configs/scrfd/scrfd_2.5g_80e_asr_jsar_paper_sr12.py \
  --out-dir results/compare_2.5g_paper_sr12/export_bundle
```

File tải về:

- `results/compare_2.5g_paper_sr12/export_bundle/analysis_artifacts_bundle.zip`

## 13. Nếu Colab bị ngắt

Bạn có thể resume train bằng tay:

```bash
%cd /content/insightface/detection/scrfd
!bash colab/run_in_env.sh python tools/train.py \
  configs/scrfd/scrfd_2.5g_80e_asr_jsar_paper_sr12.py \
  --work-dir work_dirs/compare_2.5g_paper_sr12/asr_jsar \
  --resume-from work_dirs/compare_2.5g_paper_sr12/asr_jsar/latest.pth
```

## 14. Tóm tắt lệnh tối thiểu

Nếu bạn chỉ muốn chạy lại từ đầu đến bảng so sánh:

```bash
%cd /content
!git clone <YOUR_FORK_URL> insightface

%cd /content/insightface
!bash detection/scrfd/colab/setup_colab_env.sh /content/insightface

%cd /content/insightface/detection/scrfd
!BASELINE_GPU=0 IMPROVED_GPU=1 \
  BATCH_SIZE_PER_GPU=32 WORKERS_PER_GPU=8 \
  BASELINE_LR=0.01 IMPROVED_LR=0.01 \
  bash colab/run_compare_2.5g_paper_sr12_baseline_vs_asr_jsar.sh
```
