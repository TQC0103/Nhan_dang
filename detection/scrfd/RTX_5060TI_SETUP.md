# RTX 5060 Ti Experimental Setup

This guide is the starting point for running SCRFD on `RTX 5060 Ti` /
Blackwell GPUs.

It does **not** migrate the repo to MMDetection 3.x. Instead, it keeps the
current SCRFD code on the MMDetection 2.x path and tries the lowest-risk port:

1. `PyTorch 2.7.1`
2. `CUDA 12.8` wheels
3. `MMCV 1.7.2` built from source
4. Local SCRFD repo installed in editable mode

This path is still experimental, but it is much smaller than a full MMEngine /
MMDetection 3.x migration.

## 1. Why The Legacy Env Does Not Work

The legacy SCRFD setup in this repo uses:

- `PyTorch 1.10.0`
- `CUDA 11.3`
- `mmcv-full 1.4.0`

That stack is too old for `RTX 5060 Ti`. The symptom is usually:

```text
CUDA capability sm_120 is not compatible with the current PyTorch installation
```

## 2. What This Experimental Setup Changes

- uses a separate env: `scrfd-rtx50`
- installs `torch==2.7.1`, `torchvision==0.22.1`, `torchaudio==2.7.1`
- builds `mmcv==1.7.2` from source with CUDA ops
- keeps SCRFD code in the current repo unchanged as much as possible
- uses `SCRFD_MMCV_MAX_VERSION=1.7.2` so the repo accepts the newer MMCV

## 3. Prerequisites

You need:

- an `RTX 5060 Ti`
- a recent NVIDIA driver
- build tools such as `gcc`, `g++`, `cmake`, `ninja`

Optional but preferred:

- a local CUDA toolkit with `nvcc`

Check:

```bash
nvidia-smi
nvcc --version
```

If `nvcc` is missing, the setup now falls back to `mmcv` without compiled ops
and uses repo-side fallbacks for a first smoke-test.

## 4. Setup

From the SCRFD directory:

```bash
cd detection/scrfd
bash setup_rtx50_env.sh
```

The script will:

1. create / reuse `scrfd-rtx50`
2. install PyTorch 2.7.1 CUDA 12.8 wheels
3. clone `mmcv` source into `.externals/mmcv-1.7.2`
4. build MMCV ops with `TORCH_CUDA_ARCH_LIST=12.0` when `nvcc` is available
5. install the local SCRFD package
6. run import checks

## 5. Run Commands In The New Env

Use the wrapper:

```bash
bash run_rtx50_env.sh python tools/print_config.py configs/scrfd/scrfd_500m.py
```

It injects:

```bash
SCRFD_MMCV_MAX_VERSION=1.7.2
```

which lets the repo accept the experimental MMCV version.

You can also export it manually:

```bash
export SCRFD_MMCV_MAX_VERSION=1.7.2
~/.local/bin/micromamba run -r ~/.local/micromamba -n scrfd-rtx50 <command>
```

## 6. First Smoke Test

```bash
bash run_rtx50_env.sh python -c "import torch, mmcv, mmdet; print(torch.__version__, mmcv.__version__, mmdet.__version__)"
bash run_rtx50_env.sh python tools/print_config.py configs/scrfdgen500m/scrfdgen500m_0.py
```

## 7. First Training Test

Before launching NAS search, verify one candidate trains:

```bash
export SCRFD_TRAIN_EXTRA_ARGS="--cfg-options data.workers_per_gpu=4 data.pin_memory=True data.persistent_workers=True data.prefetch_factor=4"
bash run_rtx50_env.sh python tools/train.py configs/scrfdgen500m/scrfdgen500m_0.py --no-validate
```

If that works, then move on to candidate generation and search.

## 8. What To Expect

Best case:

- MMCV 1.7.2 builds cleanly
- imports pass
- one-candidate training works

Fallback case:

- if `nvcc` is missing, the script installs plain `mmcv==1.7.2`
- repo-side fallbacks are used for `nms`, `roi_align`, and `sigmoid_focal_loss`
- this is enough for an initial SCRFD smoke-test, but not the ideal final setup

Common failure points:

- CUDA toolkit / driver mismatch
- MMCV source build breaks on your compiler / toolkit combo
- runtime incompatibilities between MMDetection 2.x code and the newer PyTorch stack

If the first training test fails after this setup, the next step is no longer
"small setup fixes". At that point, the repo likely needs a real migration
toward MMEngine / MMDetection 3.x.
