# Micromamba Env Activation

Guide này ghi lại cách activate và chạy lệnh với các env `micromamba` dùng trong repo `detection/scrfd`.

## 1. Tổng quan các env

- `scrfd-vps`
  - env mặc định do `setup_vps_env.sh` tạo
- `scrfd-colab`
  - env mặc định do `colab/setup_colab_env.sh` tạo
- `scrfd-rtx50`
  - env experimental cho GPU Blackwell / RTX 50 do `setup_rtx50_env.sh` tạo

## 2. Cách activate shell thật sự

Áp dụng cho `bash` trên Linux / VPS / Colab terminal.

### 2.1. `scrfd-vps`

Mặc định:

- `MAMBA_ROOT_PREFIX=$HOME/.local/micromamba`
- `MICROMAMBA_BIN=$HOME/.local/bin/micromamba`

Activate:

```bash
export MAMBA_ROOT_PREFIX="$HOME/.local/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook -s bash)"
micromamba activate scrfd-vps
```

### 2.2. `scrfd-colab`

Mặc định:

- `MAMBA_ROOT_PREFIX=/content/micromamba`
- `MICROMAMBA_BIN=/content/bin/micromamba`

Activate:

```bash
export MAMBA_ROOT_PREFIX="/content/micromamba"
eval "$(/content/bin/micromamba shell hook -s bash)"
micromamba activate scrfd-colab
```

### 2.3. `scrfd-rtx50`

Mặc định:

- `MAMBA_ROOT_PREFIX=$HOME/.local/micromamba`
- `MICROMAMBA_BIN=$HOME/.local/bin/micromamba`

Activate:

```bash
export MAMBA_ROOT_PREFIX="$HOME/.local/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook -s bash)"
micromamba activate scrfd-rtx50
export SCRFD_MMCV_MAX_VERSION=1.7.2
export SCRFD_TORCH_SHARING_STRATEGY=file_system
```

Hai biến cuối nên có khi chạy `scrfd-rtx50`, vì repo này vẫn bám MMDetection 2.x path cũ.

## 3. Cách chạy không cần activate shell

Đây là cách an toàn hơn trong notebook / script.

### 3.1. `scrfd-vps`

```bash
$HOME/.local/bin/micromamba run -r "$HOME/.local/micromamba" -n scrfd-vps python --version
```

### 3.2. `scrfd-colab`

```bash
/content/bin/micromamba run -r /content/micromamba -n scrfd-colab python --version
```

### 3.3. `scrfd-rtx50`

```bash
$HOME/.local/bin/micromamba run -r "$HOME/.local/micromamba" -n scrfd-rtx50 \
  env SCRFD_MMCV_MAX_VERSION=1.7.2 SCRFD_TORCH_SHARING_STRATEGY=file_system \
  python --version
```

## 4. Wrapper scripts có sẵn trong repo

Repo này đã có wrapper, thường tiện hơn activate tay.

### 4.1. Colab env

```bash
bash colab/run_in_env.sh python tools/print_config.py configs/scrfd/scrfd_500m.py
```

`colab/run_in_env.sh` mặc định dùng:

- env: `scrfd-colab`
- root: `/content/micromamba`
- micromamba: `/content/bin/micromamba`

Muốn đổi sang env khác, ví dụ `scrfd-rtx50`:

```bash
SCRFD_COLAB_ENV=scrfd-rtx50 \
MAMBA_ROOT_PREFIX=/root/.local/micromamba \
MICROMAMBA_BIN=/root/.local/bin/micromamba \
bash colab/run_in_env.sh python --version
```

### 4.2. RTX50 / Blackwell env

```bash
bash run_rtx50_env.sh python tools/print_config.py configs/scrfd/scrfd_500m.py
```

`run_rtx50_env.sh` tự inject:

- `SCRFD_MMCV_MAX_VERSION=1.7.2`
- `SCRFD_TORCH_SHARING_STRATEGY=file_system`

Nên với `scrfd-rtx50`, wrapper này là cách khuyên dùng.

## 5. Sanity check sau khi activate

```bash
python -c "import torch, mmcv, mmdet; print(torch.__version__, mmcv.__version__, mmdet.__version__)"
```

Nếu đang dùng GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 6. Kiểm tra env đang dùng

```bash
which python
python -V
echo "$CONDA_PREFIX"
micromamba env list
```

Trong notebook, nếu nghi ngờ đang rơi về Python hệ thống, ưu tiên wrapper thay vì activate tay.

## 7. Tự động activate mỗi lần mở shell

Thêm vào `~/.bashrc`:

```bash
export MAMBA_ROOT_PREFIX="$HOME/.local/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook -s bash)"
```

Sau đó:

```bash
source ~/.bashrc
micromamba activate scrfd-vps
```

Hoặc:

```bash
source ~/.bashrc
micromamba activate scrfd-rtx50
```

## 8. Deactivate

```bash
micromamba deactivate
```

## 9. Khuyến nghị thực dụng

- VPS:
  - có thể activate `scrfd-vps` trực tiếp
- Colab:
  - thường nên dùng `bash colab/run_in_env.sh ...`
- Blackwell / RTX50:
  - nên dùng `bash run_rtx50_env.sh ...`
  - hoặc nếu activate tay thì nhớ export thêm:
    - `SCRFD_MMCV_MAX_VERSION=1.7.2`
    - `SCRFD_TORCH_SHARING_STRATEGY=file_system`
