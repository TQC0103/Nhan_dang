# SCRFD Experiments Guide

## Overview

This directory contains code for three experiments on SCRFD:

1. **Baseline**: Training SCRFD 500M on WIDERFace
2. **Knowledge Distillation**: Comparing SCRFD→SCRFD vs RetinaFace→SCRFD distillation
3. **Kernel Architecture Search**: Searching kernel sizes (3x3, 5x5, 7x7) in addition to channels/depths

---

## Phase 1: Baseline Training

### Setup
```bash
# Install environment
bash setup_environment.sh

# Optional: download raw data/models
bash download_data.sh

# Prepare SCRFD data layout from an existing dataset location
bash prepare_retinaface_data.sh \
  --source-root /path/to/dataset_or_widerface_root \
  --ann-root /path/to/annotation_root \
  --force

# Or let the helper download and prepare the dataset directly
bash prepare_retinaface_data.sh \
  --download-all \
  --force
```

### Training
```bash
cd insightface/detection/scrfd

# Train SCRFD 500M baseline
bash ./tools/dist_train.sh ./configs/scrfd/scrfd_500m.py 8
```

### Expected Results
| Metric | Expected |
|--------|----------|
| Easy | ~90.57% |
| Medium | ~88.12% |
| Hard | ~68.51% |

---

## Phase 2: Knowledge Distillation Experiments

### Experiments

| Config | Teacher | Student | Purpose |
|--------|---------|---------|---------|
| `scrfd_500m_kd_10g.py` | SCRFD 10G | SCRFD 500M | Intra-SCRFD distillation |
| `scrfd_500m_kd_2.5g.py` | SCRFD 2.5G | SCRFD 500M | Intra-SCRFD (smaller teacher) |
| `scrfd_500m_kd_retinaface.py` | RetinaFace | SCRFD 500M | Cross-architecture baseline |
| `scrfd_500m_self_distill.py` | SCRFD 500M | SCRFD 500M | Self-distillation baseline |

### Hypothesis
SCRFD→SCRFD distillation (KD-1, KD-2) should outperform RetinaFace→SCRFD (KD-3) because:
- SCRFD's computation redistribution is already optimized
- The teacher and student share similar feature representations
- Knowledge transfer is more effective within the same architecture family

### Training KD Models
```bash
# SCRFD 10G -> SCRFD 500M
bash ./tools/dist_train.sh ./configs/kd/scrfd_500m_kd_10g.py 8 \
  --cfg-options model.teacher.pretrained=/path/to/scrfd_10g.pth

# SCRFD 2.5G -> SCRFD 500M
bash ./tools/dist_train.sh ./configs/kd/scrfd_500m_kd_2.5g.py 8 \
  --cfg-options model.teacher.pretrained=/path/to/scrfd_2.5g.pth

# RetinaFace -> SCRFD 500M (baseline)
bash ./tools/dist_train.sh ./configs/kd/scrfd_500m_kd_retinaface.py 8 \
  --cfg-options model.teacher.pretrained=/path/to/retinaface_teacher.pth

# Self-distillation
bash ./tools/dist_train.sh ./configs/kd/scrfd_500m_self_distill.py 8 \
  --cfg-options model.teacher.pretrained=/path/to/scrfd_500m_teacher.pth
```

### Enhanced Evaluation with Detailed Logging
```bash
# Use enhanced evaluation script for detailed logging
python ./tools/test_widerface_enhanced.py \
    configs/kd/scrfd_500m_kd_10g.py \
    work_dirs/scrfd_500m_kd_10g/latest.pth \
    --out results/kd_10g \
    --save-preds
```

---

## Logging and Analysis for Reports

### Training Logs

Each experiment produces logs in `work_dirs/<experiment_name>/`:

| File | Description |
|------|-------------|
| `<timestamp>.log` | Text log with all training iterations |
| `loss_log.csv` | CSV format loss values (if KD hooks enabled) |
| `scalars.json` | TensorBoard scalar logs |
| `epoch_*.pth` | Model checkpoints per epoch |

### Loss Components Logged (KD Experiments)

The training logs include these loss components:

**Task Losses:**
- `loss_cls` - Classification loss (Quality Focal Loss)
- `loss_bbox` - Bounding box regression loss (DIoU Loss)
- `loss_kps` - Keypoint loss (if enabled)

**Distillation Losses:**
- `loss_cls_distill` - KL divergence on classification scores
- `loss_bbox_distill` - L2 loss on bbox predictions
- `loss_distill` - Combined distillation loss

**Ratios (computed):**
- `cls_distill_ratio` - distill_cls / (task_cls + distill_cls)
- `bbox_distill_ratio` - distill_bbox / (task_bbox + distill_bbox)
- `distill_total_ratio` - total_distill / (task_total + distill_total)

### Analyzing Logs

Use the provided analysis script to generate plots:

```bash
# After training completes, use Python to analyze:
python -c "
import json
import os

# Load training log
work_dir = 'work_dirs/scrfd_500m_kd_10g'
log_file = os.path.join(work_dir, 'loss_log.csv')

# Read and analyze
with open(log_file) as f:
    lines = f.readlines()

# Parse CSV to get loss values
import csv
reader = csv.DictReader(lines)
data = list(reader)

# Plot loss curves
import matplotlib.pyplot as plt

iters = [int(row['iter']) for row in data]
loss_cls = [float(row['loss_cls']) for row in data]
loss_distill = [float(row['loss_distill']) for row in data]

plt.figure(figsize=(12, 6))
plt.plot(iters, loss_cls, label='Task Loss (cls)')
plt.plot(iters, loss_distill, label='Distill Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Curves')
plt.savefig('loss_curves.png')
plt.close()

print('Loss curves saved to loss_curves.png')
"
```

### Evaluation Results

Results saved to `results/<experiment_name>/`:

| File | Description |
|------|-------------|
| `results_summary.json` | Complete results in JSON |
| `results_summary.csv` | Easy-to-parse CSV format |
| `predictions/` | Per-image detection results |

### Expected KD Results Format

```csv
metric,value
easy_AP,0.9156
medium_AP,0.8934
hard_AP,0.7012
mAP,0.8367
config,configs/kd/scrfd_500m_kd_10g.py
checkpoint,work_dirs/scrfd_500m_kd_10g/latest.pth
```

---

## Phase 3: Kernel Architecture Search

### Overview
This kernel NAS keeps the **same two-stage SCRFD search procedure**:
1. `mode 1`: generate backbone candidates under a FLOPs budget
2. Train every candidate independently
3. Evaluate every candidate on WIDERFace
4. Select the best candidate by **Hard AP**
5. `mode 2`: search the full detector around that best template
6. Train/evaluate again and select the final model

The only intended change is the **search space**: kernel sizes are added.

The original SCRFD architecture search (`generate_configs_2.5g.py`) searches over:
- Block type (BasicBlock/Bottleneck)
- Channel widths per stage
- Stage depths (number of blocks)
- FPN/head channels and depths

**Missing**: Kernel sizes

### New Search Tool
`search_tools/generate_configs_2.5g_kernel_search.py` adds:
- Stem/first kernel search: 3x3, 5x5, or 7x7
- Stem depthwise kernel search: 3x3, 5x5, or 7x7
- Per-stage kernel sizes: 3x3, 5x5, or 7x7

### Usage

```bash
cd insightface/detection/scrfd

# Create output directory for generated configs
mkdir -p configs/scrfdgen500m_kernel

# Generate 64 backbone configs with kernel search for SCRFD 500M
python search_tools/generate_configs_2.5g_kernel_search.py \
    --mode 1 \
    --kernel-search \
    --group configs/scrfdgen500m_kernel \
    --template-config configs/scrfdgen500m/scrfdgen500m_0.py \
    --gflops 0.5 \
    --num-configs 64

# Train each config (example for first config)
bash ./tools/dist_train.sh configs/scrfdgen500m_kernel/scrfdgen500m_kernel_1.py 8

# Evaluate best config
python ./tools/test_widerface_enhanced.py \
    configs/scrfdgen500m_kernel_best.py \
    work_dirs/scrfdgen500m_kernel_best/epoch_xxx.pth \
    --out results/kernel_search_best

# Visualize all searched candidates like the original SCRFD workflow
python search_tools/visualize_search.py \
    --group configs/scrfdgen500m_kernel \
    --result-dir wouts \
    --prefix scrfdgen500m_kernel \
    --idx-from 0 \
    --idx-to 64 \
    --score-key hard \
    --topk 10
```

### Exact SCRFD-Style Workflow on 8 GPUs

Backbone search:
```bash
cd insightface/detection/scrfd

# Step 1: generate 64 backbone candidates at 0.5 GFLOPs
mkdir -p configs/scrfdgen500m_kernel
python search_tools/generate_configs_2.5g_kernel_search.py \
    --mode 1 \
    --kernel-search \
    --group configs/scrfdgen500m_kernel \
    --template-config configs/scrfdgen500m/scrfdgen500m_0.py \
    --gflops 0.5 \
    --num-configs 64

# Step 2: train all candidates, 1 GPU per candidate
bash search_tools/search_train.sh scrfdgen500m_kernel 8 8 0

# Optional: run 2 candidates concurrently on each GPU
# Keeps the SCRFD batch size unchanged, but multiplexes jobs per GPU
bash search_tools/search_train.sh scrfdgen500m_kernel 8 8 0 2 0

# Step 3: evaluate all candidates
bash search_tools/search_test_parallel.sh scrfdgen500m_kernel 8 8 0 wouts 0.02 scrfdgen500m_kernel

# Step 4: rank by Hard AP
python search_tools/visualize_search.py \
    --group configs/scrfdgen500m_kernel \
    --result-dir wouts \
    --prefix scrfdgen500m_kernel \
    --idx-from 0 \
    --idx-to 64 \
    --score-key hard \
    --topk 10
```

Full-network search around the best backbone candidate:
```bash
# Assume the best candidate after mode 1 is scrfdgen500m_kernel_17.py
mkdir -p configs/scrfdgen500m_kernel_all
cp configs/scrfdgen500m_kernel/scrfdgen500m_kernel_17.py \
   configs/scrfdgen500m_kernel_all/scrfdgen500m_kernel_all_0.py

python search_tools/generate_configs_2.5g_kernel_search.py \
    --mode 2 \
    --kernel-search \
    --group configs/scrfdgen500m_kernel_all \
    --template-config configs/scrfdgen500m_kernel_all/scrfdgen500m_kernel_all_0.py \
    --gflops 0.5 \
    --num-configs 64
```

### Kernel-Only Search From The Searched SCRFD 500M

If you want to keep the searched SCRFD 500M architecture and only search
kernels, use:

```bash
bash search_tools/generate_scrfd500m_kernel_only.sh 16
```

```bash
bash search_tools/generate_scrfd500m_kernel_only.sh 32
```

Equivalent explicit command:

```bash
python search_tools/parallel_generate.py \
    --group configs/scrfd500m_kernel_only \
    --template-config configs/scrfdgen500m/scrfd500m_kernel_seed.py \
    --mode 1 \
    --kernel-search \
    --kernel-only \
    --gflops 0.5 \
    --num-configs 16 \
    --workers 8 \
    --oversample-factor 2.0 \
    --keep-workdir
```

### Files Created
- `mmdet/models/backbones/mobilenet_v1_ks.py`: MobileNetV1 with searchable stem/first kernel and stage kernels
- `search_tools/generate_configs_2.5g_kernel_search.py`: Extended search tool with kernel search
- `search_tools/visualize_search.py`: Plot AP/FLOPs/compute-distribution/kernel-distribution for searched models
- `configs/scrfdgen500m/scrfdgen500m_0.py`: Template config for 500M kernel search

---

## KD Implementation Details

### Files
- `mmdet/core/distillation/kd_losses.py`: KL and L2 distillation losses
- `mmdet/core/distillation/logging.py`: Custom logging hooks for KD
- `mmdet/models/detectors/scrfd_kd.py`: SCRFD detector with KD support
- `configs/kd/*.py`: KD experiment configs

### Distillation Losses

**KL Divergence on Classification Scores:**
```python
cls_loss = KL(softmax(s_cls/T), softmax(t_cls/T)) * T^2
```

**L2 Loss on Bbox Predictions:**
```python
bbox_loss = MSE(s_bbox, t_bbox)
```

### Combined Loss
```python
loss_distill = cls_weight * cls_loss + bbox_weight * bbox_loss
loss_total = loss_task + loss_distill
```

---

## Pretrained Models (for KD teachers)

Download from OneDrive links in main README:
- SCRFD 500M: https://1drv.ms/u/s!AswpsDO2toNKqyYWxScdiTITY4TQ
- SCRFD 2.5G: https://1drv.ms/u/s!AswpsDO2toNKqyTIXnzB1ujPq4th
- SCRFD 10G: https://1drv.ms/u/s!AswpsDO2toNKqyUKwTiwXv2kaa8o

---

## Expected Timeline

| Phase | Task | Time |
|-------|------|------|
| 1 | Baseline training | ~6-8 hours (8 GPUs) |
| 2 | KD experiments | ~4 experiments × 6-8 hours |
| 3 | Kernel search | 64 configs × 1 epoch quick test + full training of top configs |

---

## Code Quality Checks

### Before running, verify:

1. **Import check:**
```bash
cd insightface/detection/scrfd
python -c "from mmdet.models.detectors.scrfd_kd import SCRFDKD; print('SCRFDKD imported OK')"
python -c "from mmdet.core.distillation import KLDistillLoss; print('KD losses imported OK')"
```

2. **Config check:**
```bash
python tools/train.py configs/kd/scrfd_500m_kd_10g.py --no-validate \
  --cfg-options model.teacher.pretrained=/path/to/scrfd_10g.pth
```

This should load the config without errors before actual training.

### Common Issues

1. **ImportError**: Ensure mmcv and mmdet are properly installed
2. **CUDA out of memory**: Reduce batch size in config
3. **Dataset not found**: Check data paths in config
4. **Teacher model mismatch**: Ensure teacher architecture matches expected outputs
