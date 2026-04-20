# VPS Guide: Environment, Dataset, Search, And ASR+JSAR

Tài liệu này gom lại một luồng chạy thực dụng trên VPS cho SCRFD:

- cài môi trường bằng `setup_vps_env.sh`
- kích hoạt env `scrfd-vps`
- tải dataset WIDERFace / RetinaFace layout bằng `prepare_retinaface_data.sh`
- sinh config tìm kiếm cho các mức FLOPs khác nhau
- train / test các config đã sinh
- chạy các config cải tiến `ASR+JSAR`

Tất cả lệnh dưới đây giả định bạn đang đứng trong:

```bash
cd /path/to/insightface/detection/scrfd
```

## 1. Cài môi trường VPS bằng `setup_vps_env.sh`

Script [setup_vps_env.sh](../setup_vps_env.sh) sẽ:

- cài `micromamba`
- tạo env `scrfd-vps`
- cài `PyTorch 1.10.0 + CUDA 11.3`
- cài `mmcv-full 1.4.0`
- cài package local của repo này

Chạy mặc định:

```bash
bash setup_vps_env.sh
```

Nếu VPS đã có sẵn các gói hệ thống cần thiết và bạn không muốn script chạy `apt`:

```bash
SCRFD_INSTALL_SYSTEM_DEPS=0 bash setup_vps_env.sh
```

Một số biến môi trường hữu ích:

```bash
SCRFD_ENV_NAME=scrfd-vps
MAMBA_ROOT_PREFIX="$HOME/.local/micromamba"
MICROMAMBA_BIN="$HOME/.local/bin/micromamba"
```

Sau khi script chạy xong, repo sẽ in ra ví dụ chạy lệnh trong env bằng `micromamba run`.

## 2. Activate env `scrfd-vps`

Cách activate trong `bash`:

```bash
export MAMBA_ROOT_PREFIX="$HOME/.local/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook -s bash)"
micromamba activate scrfd-vps
```

Kiểm tra nhanh:

```bash
which python
python -V
python -c "import torch, mmcv, mmdet; print(torch.__version__, mmcv.__version__, mmdet.__version__)"
```

Nếu không muốn activate hẳn shell, dùng:

```bash
$HOME/.local/bin/micromamba run -r "$HOME/.local/micromamba" -n scrfd-vps python --version
```

## 3. Tải dataset bằng `prepare_retinaface_data.sh --download-all`

Script [prepare_retinaface_data.sh](../prepare_retinaface_data.sh) sẽ:

- tải `WIDER_train.zip`, `WIDER_val.zip`, `wider_face_split.zip`
- tải annotation bundle SCRFD
- tự dựng layout dữ liệu mà SCRFD mong đợi:

```text
data/retinaface/
  train/
    images/
    labelv2.txt
  val/
    images/
    labelv2.txt
    gt/
```

Chạy mặc định:

```bash
bash prepare_retinaface_data.sh --download-all --force
```

Nếu không muốn ghi cache download vào chỗ mặc định:

```bash
bash prepare_retinaface_data.sh \
  --download-all \
  --download-root /path/to/download_cache \
  --force
```

Kiểm tra nhanh sau khi xong:

```bash
find -L data/retinaface/train/images -type f | wc -l
find -L data/retinaface/val/images -type f | wc -l
ls data/retinaface/train/labelv2.txt
ls data/retinaface/val/gt | head
```

## 4. Generate config nguyên bản bằng script gốc

Phần này chỉ dùng script gốc [generate_configs_2.5g.py](../search_tools/generate_configs_2.5g.py).

Không dùng:

- `kernel search`
- `parallel_generate.py`
- các wrapper mở rộng

Lưu ý quan trọng:

- script gốc này phù hợp với flow search nguyên bản của SCRFD cho các cấu hình kiểu ResNet
- trong repo hiện tại, phần đó tương ứng thực tế với các mức như `1.0G`, `2.5G`, `10G`, `34G`
- `500M` trong repo này thuộc MobileNet family; để search `500M` đúng kiểu MobileNet, repo đang dùng nhánh mở rộng có `kernel search`

Vì vậy, trong mục này:

- tôi chỉ hướng dẫn generate config nguyên bản cho `1.0G`, `2.5G`, `10G`, `34G`
- `500M` sẽ không đưa vào phần generator gốc để tránh nhầm với flow MobileNet mở rộng

### 4.1. Quy ước đặt tên group config

Script gốc dùng quy ước:

- thư mục config group nằm trong `configs/<group_name>`
- file seed đầu tiên phải có dạng:

```text
configs/<group_name>/<group_name>_0.py
```

Sau khi generate, script sẽ tiếp tục tạo:

```text
configs/<group_name>/<group_name>_1.py
configs/<group_name>/<group_name>_2.py
...
```

Script train/test search cũng dùng đúng `group_name` làm prefix, nên hãy luôn đặt tên thư mục và tên file seed theo cùng một prefix.

### 4.2. Seed configs gợi ý

| Mức FLOPs | Seed config gợi ý | Ghi chú |
| --- | --- | --- |
| 1.0G | `configs/scrfd/base_1g.py` | ResNet family |
| 2.5G | `configs/scrfdgen2.5g/scrfdgen2.5g_0.py` hoặc `configs/scrfd/base_2.5g.py` | ResNet family |
| 10G | `configs/scrfd/base_10g.py` | ResNet family |
| 34G | `configs/scrfd/base_34g.py` | ResNet family |

### 4.3. Ví dụ generate cho từng phiên bản

#### 1.0G

```bash
mkdir -p configs/scrfdgen1g
cp configs/scrfd/base_1g.py configs/scrfdgen1g/scrfdgen1g_0.py

python search_tools/generate_configs_2.5g.py \
  --group configs/scrfdgen1g \
  --gflops 1.0 \
  --num-configs 16
```

#### 2.5G

```bash
mkdir -p configs/scrfdgen2.5g_search
cp configs/scrfdgen2.5g/scrfdgen2.5g_0.py \
   configs/scrfdgen2.5g_search/scrfdgen2.5g_search_0.py

python search_tools/generate_configs_2.5g.py \
  --group configs/scrfdgen2.5g_search \
  --gflops 2.5 \
  --num-configs 16
```

#### 10G

```bash
mkdir -p configs/scrfdgen10g
cp configs/scrfd/base_10g.py configs/scrfdgen10g/scrfdgen10g_0.py

python search_tools/generate_configs_2.5g.py \
  --group configs/scrfdgen10g \
  --gflops 10.0 \
  --num-configs 16
```

#### 34G

```bash
mkdir -p configs/scrfdgen34g
cp configs/scrfd/base_34g.py configs/scrfdgen34g/scrfdgen34g_0.py

python search_tools/generate_configs_2.5g.py \
  --group configs/scrfdgen34g \
  --gflops 34.0 \
  --num-configs 16
```

Lưu ý:

- script gốc mặc định sẽ dùng file `<group_name>_0.py` làm template
- vì vậy chỉ cần copy đúng file seed vào group rồi chạy lệnh trên
- tài liệu này không dùng kernel search và cũng không trình bày flow mở rộng khác

## 5. Chạy search từ generated configs

### 5.1. Train toàn bộ generated configs

Script [search_train.sh](../search_tools/search_train.sh) nhận:

```text
bash search_tools/search_train.sh <group> <gpus> <tasks_per_gpu> <offset> <candidates_per_gpu> <use_dist> <port_base>
```

Ví dụ, nếu group `scrfdgen2.5g_search` có:

```text
scrfdgen2.5g_search_0.py   # seed
scrfdgen2.5g_search_1.py
...
scrfdgen2.5g_search_16.py
```

và bạn muốn train 16 config sinh ra trên 2 GPU:

```bash
bash search_tools/search_train.sh scrfdgen2.5g_search 2 8 1 1 1 29100
```

Giải thích:

- `2`: số GPU
- `8`: mỗi GPU xử lý 8 candidate
- `1`: bắt đầu từ config `_1.py`, bỏ qua seed `_0.py`
- `1`: mỗi GPU chạy 1 candidate tại một thời điểm
- `1`: dùng `dist_train.sh`
- `29100`: base port

Nếu muốn chạy không dùng distributed launcher:

```bash
bash search_tools/search_train.sh scrfdgen2.5g_search 2 8 1 1 0 29100
```

Log mặc định:

```text
gpu0_slot0.log
gpu1_slot0.log
...
```

### 5.2. Test toàn bộ generated configs

```bash
bash search_tools/search_test_parallel.sh scrfdgen2.5g_search 2 8 1 wouts 0.02 scrfdgen2.5g_search
```

Kết quả sẽ nằm dưới:

```text
wouts/scrfdgen2.5g_search/scrfdgen2.5g_search_1/
wouts/scrfdgen2.5g_search/scrfdgen2.5g_search_2/
...
```

### 5.3. Vẽ biểu đồ search giống paper

Script [visualize_search.py](../search_tools/visualize_search.py) sẽ tổng hợp:

- AP theo candidate
- AP theo FLOPs
- top-k summary
- heatmap / compute distribution nếu metadata có sẵn

Ví dụ:

```bash
python search_tools/visualize_search.py \
  --group configs/scrfdgen2.5g_search \
  --result-dir wouts \
  --prefix scrfdgen2.5g_search \
  --idx-from 1 \
  --idx-to 17 \
  --topk 10 \
  --score-key hard \
  --output-dir results/scrfdgen2.5g_search_viz
```

## 6. Convenience script cho search 500M

Nếu mục tiêu của bạn là:

- 500M
- 16 configs
- train/test/visualize/package luôn

thì có thể dùng wrapper:

```bash
bash run_search_500m_16configs_bundle.sh
```

Script [run_search_500m_16configs_bundle.sh](../run_search_500m_16configs_bundle.sh) sẽ:

- generate configs
- train các candidate
- test các candidate
- vẽ biểu đồ search
- đóng gói toàn bộ artifact thành bundle zip để tải về

Nếu bạn chỉ muốn flow nguyên bản, có thể bỏ qua script này và dùng trực tiếp các lệnh ở Mục 4 và Mục 5.

Nếu bạn cần search `500M` trong repo hiện tại, đây là ngoại lệ:

- không đi theo generator gốc `generate_configs_2.5g.py`
- phải dùng flow mở rộng dành cho MobileNet / kernel-search
- phần đó nằm ở:
  - [run_search_500m_16configs_bundle.sh](../run_search_500m_16configs_bundle.sh)
  - [EXPERIMENTS_README.md](../EXPERIMENTS_README.md)

## 7. Chạy cải tiến ASR+JSAR

### 7.1. Các config có sẵn trong repo

Hiện repo đã có sẵn các config cải tiến:

- `configs/scrfd/scrfd_500m_asr.py`
- `configs/scrfd/scrfd_500m_asr_jsar.py`
- `configs/scrfd/scrfd_2.5g_80e_asr_jsar.py`
- `configs/scrfd/scrfd_2.5g_80e_asr_jsar_paper_sr12.py`
- `configs/scrfd/scrfd_2.5g_80e_cosine_asr_jsar.py`
- `configs/scrfd/scrfd_2.5g_80e_cosine_asr_jsar_paper_sr12.py`

Hiện chưa có file `ASR+JSAR` dựng sẵn cho `1G`, `10G`, `34G`. Nếu cần, bạn có thể copy pattern từ:

- `scrfd_500m_asr_jsar.py`
- `scrfd_2.5g_80e_asr_jsar.py`

### 7.2. Train trực tiếp

Ví dụ với 500M:

```bash
python tools/train.py configs/scrfd/scrfd_500m_asr_jsar.py
```

Ví dụ với 2.5G:

```bash
python tools/train.py configs/scrfd/scrfd_2.5g_80e_asr_jsar.py
```

Nếu có nhiều GPU:

```bash
bash tools/dist_train.sh configs/scrfd/scrfd_2.5g_80e_asr_jsar.py 2
```

### 7.3. Evaluate

```bash
python tools/test_widerface_enhanced.py \
  configs/scrfd/scrfd_2.5g_80e_asr_jsar.py \
  work_dirs/scrfd_2.5g_80e_asr_jsar/latest.pth \
  --out results/scrfd_2.5g_80e_asr_jsar \
  --save-preds
```

### 7.4. Wrapper compare baseline vs ASR+JSAR

Trên VPS có sẵn wrapper cho `2.5G paper SR12`:

```bash
bash run_compare_2.5g_paper_sr12_baseline_vs_asr_jsar.sh
```

Wrapper này sẽ:

- train baseline
- train ASR+JSAR
- evaluate
- tạo comparison
- package artifact bundle để tải về

Nếu muốn dùng GPU/batch size/workers cụ thể:

```bash
BASELINE_GPU=0 IMPROVED_GPU=1 \
BATCH_SIZE_PER_GPU=32 WORKERS_PER_GPU=8 \
BASELINE_LR=0.02 IMPROVED_LR=0.02 \
PIN_MEMORY=1 PERSISTENT_WORKERS=1 PREFETCH_FACTOR=4 \
bash run_compare_2.5g_paper_sr12_baseline_vs_asr_jsar.sh
```

## 8. Gợi ý luồng chạy thực tế trên VPS

### Luồng 1: train baseline / ASR+JSAR

```bash
bash setup_vps_env.sh

export MAMBA_ROOT_PREFIX="$HOME/.local/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook -s bash)"
micromamba activate scrfd-vps

bash prepare_retinaface_data.sh --download-all --force

python tools/train.py configs/scrfd/scrfd_500m_asr_jsar.py
```

### Luồng 2: search một phiên bản mới

```bash
bash setup_vps_env.sh

export MAMBA_ROOT_PREFIX="$HOME/.local/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook -s bash)"
micromamba activate scrfd-vps

bash prepare_retinaface_data.sh --download-all --force

mkdir -p configs/scrfdgen1g
cp configs/scrfd/base_1g.py configs/scrfdgen1g/scrfdgen1g_0.py

python search_tools/generate_configs_2.5g_kernel_search.py \
  --group configs/scrfdgen1g \
  --template-config configs/scrfdgen1g/scrfdgen1g_0.py \
  --mode 1 \
  --gflops 1.0 \
  --num-configs 16

bash search_tools/search_train.sh scrfdgen1g 2 8 1 1 1 29100

bash search_tools/search_test_parallel.sh scrfdgen1g 2 8 1 wouts 0.02 scrfdgen1g

python search_tools/visualize_search.py \
  --group configs/scrfdgen1g \
  --result-dir wouts \
  --prefix scrfdgen1g \
  --idx-from 1 \
  --idx-to 17 \
  --score-key hard \
  --output-dir results/scrfdgen1g_viz
```

## 9. Tài liệu liên quan

- [README.md](../README.md)
- [EXPERIMENTS_README.md](../EXPERIMENTS_README.md)
- [asr_jsar.md](asr_jsar.md)
- [micromamba_env_activation.md](micromamba_env_activation.md)
- [hard_gain_analysis_end_to_end.md](hard_gain_analysis_end_to_end.md)
