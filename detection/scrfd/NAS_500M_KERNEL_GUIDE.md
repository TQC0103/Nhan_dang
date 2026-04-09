# SCRFD 500M Kernel NAS Guide

This guide describes how to run SCRFD 500M kernel NAS while keeping the
original SCRFD search procedure unchanged.

The search flow is still:

1. Generate backbone candidates under a FLOPs budget
2. Train every candidate independently
3. Evaluate every candidate on WIDERFace
4. Select the best candidate by Hard AP
5. Use that candidate as the template for full-network search
6. Train/evaluate again and select the final model

The only search-space change is that kernel sizes are also searched.

## 1. Hardware Assumption

This guide targets:

- 8x V100 32GB
- 1 GPU per candidate during search
- WIDERFace / RetinaFace-style dataset layout used by SCRFD

## 2. Important Files

- `search_tools/generate_configs_2.5g_kernel_search.py`
- `search_tools/search_train.py`
- `search_tools/search_train.sh`
- `search_tools/search_test.sh`
- `search_tools/search_test_parallel.sh`
- `search_tools/visualize_search.py`
- `configs/scrfdgen500m/scrfdgen500m_0.py`

## 3. Clone The Repo

```bash
git clone https://github.com/giahuytran4205/insightface-scrfd-kd-nas-colab-20260409143017.git
cd insightface-scrfd-kd-nas-colab-20260409143017/detection/scrfd
```

If you already have the repo locally, switch to the branch that contains the
kernel NAS changes.

## 4. Environment Setup

Use a dedicated Python 3.8 environment.

```bash
conda create -n scrfd python=3.8 -y
conda activate scrfd
```

Install PyTorch and CUDA-matching packages.

```bash
pip install torch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/torch1.10.0/cu113/index.html
```

Install SCRFD dependencies and the local package.

```bash
pip install -r requirements/build.txt
pip install -r requirements/runtime.txt
pip install scipy Pillow opencv-python onnxruntime-gpu tensorboard
pip install -v -e .
```

The NAS generator now uses the local
`search_tools/search_space.py` helper, so no separate
`autotorch` / `ConfigSpace` install is required.

## 5. Prepare Dataset

The expected layout is:

```text
detection/scrfd/data/retinaface/
    train/
        images/
        labelv2.txt
    val/
        images/
        labelv2.txt
        gt/
            *.mat
```

This is the same dataset layout used by the original SCRFD training code.

If your dataset already exists somewhere else on the VPS, use the helper
script to create the expected layout under `data/retinaface`:

```bash
bash prepare_retinaface_data.sh \
  --source-root /path/to/dataset_or_widerface_root \
  --ann-root /path/to/annotation_root \
  --force
```

If your source is already in the prepared SCRFD layout, `--ann-root` can be
omitted:

```bash
bash prepare_retinaface_data.sh \
  --source-root /path/to/retinaface_prepared \
  --force
```

If the VPS does not have the dataset yet, download and prepare it directly:

```bash
bash prepare_retinaface_data.sh \
  --download-all \
  --force
```

## 6. Sanity Check

Run a few import/config checks before launching the search.

```bash
python -c "from mmdet.models.backbones.mobilenet_v1_ks import MobileNetV1KS; print('MobileNetV1KS OK')"
python -c "from mmdet.models.detectors.scrfd import SCRFD; print('SCRFD OK')"
python tools/print_config.py configs/scrfdgen500m/scrfdgen500m_0.py
```

## 7. Search Space

For SCRFD 500M kernel NAS, the search adds:

- `stem_kernel_size`
- `stem_dw_kernel_size`
- `stage_kernel_sizes` for the 4 backbone stages

The original SCRFD-style dimensions are still searched:

- backbone widths / planes
- stage depths / blocks
- in `mode 2`, neck/head widths and head depth

## 8. Stage 1: Backbone Search

Generate 64 backbone candidates at the 0.5 GFLOPs budget.

```bash
mkdir -p configs/scrfdgen500m_kernel

python search_tools/generate_configs_2.5g_kernel_search.py \
  --mode 1 \
  --kernel-search \
  --group configs/scrfdgen500m_kernel \
  --template-config configs/scrfdgen500m/scrfdgen500m_0.py \
  --gflops 0.5 \
  --num-configs 64
```

Notes:

- This creates `configs/scrfdgen500m_kernel/scrfdgen500m_kernel_0.py` to
  `scrfdgen500m_kernel_63.py` if the folder was empty.
- The generator only keeps configs whose FLOPs are close to the target.

### Parallel Candidate Generation

Config generation is CPU-bound. It does not use your 8 GPUs directly.

To speed it up on a VPS, use the parallel wrapper, which launches multiple
generator workers in parallel and merges the unique configs back into the
target group:

```bash
python search_tools/parallel_generate.py \
  --group configs/scrfdgen500m_kernel \
  --template-config configs/scrfdgen500m/scrfdgen500m_0.py \
  --mode 1 \
  --kernel-search \
  --gflops 0.5 \
  --num-configs 64 \
  --workers 8 \
  --oversample-factor 2.0
```

Notes:

- `--workers` should follow CPU availability, not GPU count.
- `--oversample-factor` helps offset duplicate/rejected configs during merge.
- The final merged configs still use the same naming format, so downstream
  `search_train.sh` and `search_test_parallel.sh` do not change.

## 9. Stage 1: Train All Backbone Candidates

This follows the original SCRFD pattern: 1 GPU trains 1 candidate at a time,
and 8 GPUs run 8 candidates in parallel.

```bash
bash search_tools/search_train.sh scrfdgen500m_kernel 8 8 0
```

Meaning:

- group = `scrfdgen500m_kernel`
- GPUs = `8`
- tasks per GPU = `8`
- offset = `0`

So the script covers:

- GPU 0: configs `0..7`
- GPU 1: configs `8..15`
- ...
- GPU 7: configs `56..63`

Logs are written to:

```text
gpu0.log
gpu1.log
...
gpu7.log
```

### Run Multiple Candidates On One GPU

The launcher now supports `candidates-per-gpu`.

Arguments:

```text
bash search_tools/search_train.sh <group> <gpus> <tasks_per_gpu> <offset> <candidates_per_gpu> <use_dist> <port_base>
```

Defaults:

- `candidates_per_gpu=1`
- `use_dist=1`
- `port_base=29100`

Example: run 2 candidates concurrently on each GPU while keeping the default
SCRFD batch size unchanged:

```bash
bash search_tools/search_train.sh scrfdgen500m_kernel 8 8 0 2 0
```

Recommended setting when running multiple candidates on one GPU:

- set `use_dist=0`
- this uses `tools/train.py` directly instead of `dist_train.sh`
- it avoids unnecessary distributed-launch overhead for single-GPU jobs

Example layouts:

- `bash search_tools/search_train.sh scrfdgen500m_kernel 8 8 0 1 1`
  one candidate per GPU, original SCRFD-style launcher
- `bash search_tools/search_train.sh scrfdgen500m_kernel 8 8 0 2 0`
  two concurrent candidates per GPU
- `bash search_tools/search_train.sh scrfdgen500m_kernel 8 8 0 3 0`
  three concurrent candidates per GPU

Each slot gets interleaved candidate indices, so the work is not duplicated.

Check training progress with:

```bash
tail -f gpu0.log
```

## 10. Stage 1: Evaluate All Backbone Candidates

After training completes, evaluate all candidates on WIDERFace.

```bash
bash search_tools/search_test_parallel.sh scrfdgen500m_kernel 8 8 0 wouts 0.02 scrfdgen500m_kernel
```

This writes outputs like:

```text
wouts/scrfdgen500m_kernel/scrfdgen500m_kernel_0/aps
wouts/scrfdgen500m_kernel/scrfdgen500m_kernel_1/aps
...
```

Each `aps` file stores:

- Easy AP
- Medium AP
- Hard AP

## 11. Stage 1: Visualize And Choose The Best Backbone Candidate

Rank all candidates using Hard AP, like the original SCRFD search procedure.

```bash
python search_tools/visualize_search.py \
  --group configs/scrfdgen500m_kernel \
  --result-dir wouts \
  --prefix scrfdgen500m_kernel \
  --idx-from 0 \
  --idx-to 64 \
  --score-key hard \
  --topk 10
```

This creates:

- `wouts/scrfdgen500m_kernel_viz/search_stats.jsonl`
- `wouts/scrfdgen500m_kernel_viz/search_stats.csv`
- `wouts/scrfdgen500m_kernel_viz/ap_vs_candidate.png`
- `wouts/scrfdgen500m_kernel_viz/ap_vs_flops.png`
- `wouts/scrfdgen500m_kernel_viz/topk_compute_distribution.png`
- `wouts/scrfdgen500m_kernel_viz/topk_kernel_heatmap.png`
- `wouts/scrfdgen500m_kernel_viz/topk_summary.md`

Pick the best model from `topk_summary.md`.

Assume the best backbone candidate is:

```text
scrfdgen500m_kernel_17
```

## 12. Stage 2: Full-Network Search

Create a new search group and use the best stage-1 config as the template.

```bash
mkdir -p configs/scrfdgen500m_kernel_all
cp configs/scrfdgen500m_kernel/scrfdgen500m_kernel_17.py \
   configs/scrfdgen500m_kernel_all/scrfdgen500m_kernel_all_0.py
```

Now generate full-network candidates around that template:

```bash
python search_tools/generate_configs_2.5g_kernel_search.py \
  --mode 2 \
  --kernel-search \
  --group configs/scrfdgen500m_kernel_all \
  --template-config configs/scrfdgen500m_kernel_all/scrfdgen500m_kernel_all_0.py \
  --gflops 0.5 \
  --num-configs 64
```

`mode 2` still follows SCRFD style. The difference is that it now also searches:

- neck output channels
- head channels
- head stacked conv count

while keeping the selected stage-1 backbone as the template center.

## 13. Stage 2: Train All Full-Network Candidates

```bash
bash search_tools/search_train.sh scrfdgen500m_kernel_all 8 8 0
```

## 14. Stage 2: Evaluate All Full-Network Candidates

```bash
bash search_tools/search_test_parallel.sh scrfdgen500m_kernel_all 8 8 0 wouts 0.02 scrfdgen500m_kernel_all
```

## 15. Stage 2: Visualize And Select The Final Model

```bash
python search_tools/visualize_search.py \
  --group configs/scrfdgen500m_kernel_all \
  --result-dir wouts \
  --prefix scrfdgen500m_kernel_all \
  --idx-from 0 \
  --idx-to 64 \
  --score-key hard \
  --topk 10
```

The best entry in:

```text
wouts/scrfdgen500m_kernel_all_viz/topk_summary.md
```

is your final selected NAS result.

## 16. Optional: Retrain The Final Best Config Cleanly

After selecting the final best config, it is recommended to retrain it once
cleanly as the final report model.

Example:

```bash
bash tools/dist_train.sh configs/scrfdgen500m_kernel_all/scrfdgen500m_kernel_all_23.py 8
```

Then evaluate it again:

```bash
python tools/test_widerface.py \
  configs/scrfdgen500m_kernel_all/scrfdgen500m_kernel_all_23.py \
  work_dirs/scrfdgen500m_kernel_all_23/latest.pth \
  --mode 0 \
  --thr 0.02 \
  --out wouts/final_scrfdgen500m_kernel_all_23
```

## 17. Expected Runtime Pattern On 8x V100 32GB

This search is still the original SCRFD-style candidate search. It is not a
supernet NAS.

Because the target model is 500M instead of 2.5G:

- each candidate is cheaper than the original 2.5G SCRFD demo search
- the search space is somewhat larger because kernels are added
- overall cost is usually still in the same ballpark as SCRFD search, and
  often lower per candidate

## 18. Troubleshooting

If `mmcv` import fails:

```bash
python -c "import mmcv; print(mmcv.__version__)"
```

If a config fails to train, inspect:

```bash
tail -n 200 gpu0.log
```

If evaluation results are missing, check whether `latest.pth` exists:

```bash
ls work_dirs/scrfdgen500m_kernel_0
```

If visualize finds no records, confirm each candidate has:

```text
wouts/<group>/<candidate>/aps
```

## 19. Minimal Command Summary

```bash
# Stage 1
python search_tools/generate_configs_2.5g_kernel_search.py --mode 1 --kernel-search --group configs/scrfdgen500m_kernel --template-config configs/scrfdgen500m/scrfdgen500m_0.py --gflops 0.5 --num-configs 64
bash search_tools/search_train.sh scrfdgen500m_kernel 8 8 0
bash search_tools/search_test_parallel.sh scrfdgen500m_kernel 8 8 0 wouts 0.02 scrfdgen500m_kernel
python search_tools/visualize_search.py --group configs/scrfdgen500m_kernel --result-dir wouts --prefix scrfdgen500m_kernel --idx-from 0 --idx-to 64 --score-key hard --topk 10

# Stage 2
cp configs/scrfdgen500m_kernel/<best>.py configs/scrfdgen500m_kernel_all/scrfdgen500m_kernel_all_0.py
python search_tools/generate_configs_2.5g_kernel_search.py --mode 2 --kernel-search --group configs/scrfdgen500m_kernel_all --template-config configs/scrfdgen500m_kernel_all/scrfdgen500m_kernel_all_0.py --gflops 0.5 --num-configs 64
bash search_tools/search_train.sh scrfdgen500m_kernel_all 8 8 0
bash search_tools/search_test_parallel.sh scrfdgen500m_kernel_all 8 8 0 wouts 0.02 scrfdgen500m_kernel_all
python search_tools/visualize_search.py --group configs/scrfdgen500m_kernel_all --result-dir wouts --prefix scrfdgen500m_kernel_all --idx-from 0 --idx-to 64 --score-key hard --topk 10
```
