# Colab Setup

Colab runtime hiện tại đã chuyển sang Python 3.11/3.12 và PyTorch 2.6+ trở lên, trong khi repo `detection/scrfd` này vẫn bám nhánh MMDetection 2.x cũ và `mmcv-full` 1.x. Vì vậy cách ổn định nhất là tạo một môi trường riêng bằng `micromamba`, thay vì cài trực tiếp vào Python mặc định của Colab.

## 1. Mount Drive nếu bạn muốn lưu checkpoint

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2. Clone repo

```bash
%cd /content
!git clone https://github.com/deepinsight/insightface.git
```

Nếu bạn dùng fork riêng:

```bash
%cd /content
!git clone <YOUR_FORK_URL> insightface
```

## 3. Setup môi trường và dataset mặc định

```bash
%cd /content/insightface
!bash detection/scrfd/colab/setup_colab_env.sh /content/insightface
```

Lệnh trên mặc định sẽ:

- tạo env `micromamba`
- cài dependency cho SCRFD
- tải WIDERFace + annotation bundle
- chuẩn bị layout `detection/scrfd/data/retinaface`

Nếu bạn muốn tắt phần dataset:

```bash
%cd /content/insightface
!SCRFD_PREPARE_RETINAFACE=0 SCRFD_DOWNLOAD_RETINAFACE=0 \
  bash detection/scrfd/colab/setup_colab_env.sh /content/insightface
```

## 4. Chạy lệnh trong env

```bash
%cd /content/insightface
!bash detection/scrfd/colab/run_in_env.sh python detection/scrfd/tools/print_config.py detection/scrfd/configs/scrfd/scrfd_500m.py
```

## 5. Train baseline SCRFD 500M

```bash
%cd /content/insightface
!bash detection/scrfd/colab/run_in_env.sh bash detection/scrfd/tools/dist_train.sh detection/scrfd/configs/scrfd/scrfd_500m.py 1
```

## 6. Train KD

Ví dụ với teacher SCRFD 10G:

```bash
%cd /content/insightface
!bash detection/scrfd/colab/run_in_env.sh bash detection/scrfd/tools/dist_train.sh \
  detection/scrfd/configs/kd/scrfd_500m_kd_10g.py \
  1 \
  --cfg-options model.teacher.pretrained=/content/drive/MyDrive/checkpoints/scrfd_10g.pth
```

## 7. NAS cho cả first kernel trên 500M

```bash
%cd /content/insightface
!mkdir -p detection/scrfd/configs/scrfdgen500m_kernel
!bash detection/scrfd/colab/run_in_env.sh python detection/scrfd/search_tools/generate_configs_2.5g_kernel_search.py \
  --mode 1 \
  --kernel-search \
  --group detection/scrfd/configs/scrfdgen500m_kernel \
  --template-config detection/scrfd/configs/scrfdgen500m/scrfdgen500m_0.py \
  --gflops 0.5 \
  --num-configs 16
```

## 8. Notes

- Nếu Colab cấp CPU runtime, setup vẫn chạy được nhưng train sẽ rất chậm.
- Nếu bạn mount Drive và làm việc trực tiếp trong `/content/drive/...`, I/O sẽ chậm hơn so với `/content`.
- Với các file lớn như dataset/checkpoint, nên copy từ Drive sang `/content` trước khi train.

## 9. Quick Candidate Search Trên Colab

Script quick search mặc định dùng 1 GPU Colab để:

- sinh một số candidate nhỏ
- train nhanh từng candidate
- test WIDERFace
- visualize và chọn best candidate

Chạy mặc định:

```bash
%cd /content/insightface/detection/scrfd
!bash colab/run_quick_search.sh all
```

Mặc định hiện tại:

- `SCRFD_COLAB_SEARCH_KIND=kernel_only`
- `SCRFD_COLAB_NUM_CONFIGS=4`
- `SCRFD_COLAB_SEARCH_EPOCHS=2`

Ví dụ đổi sang backbone search và tăng số candidate:

```bash
%cd /content/insightface/detection/scrfd
!SCRFD_COLAB_SEARCH_KIND=backbone SCRFD_COLAB_NUM_CONFIGS=8 SCRFD_COLAB_SEARCH_EPOCHS=4 \
  bash colab/run_quick_search.sh all
```

Chạy từng bước riêng:

```bash
%cd /content/insightface/detection/scrfd
!bash colab/run_quick_search.sh generate
!bash colab/run_quick_search.sh train
!bash colab/run_quick_search.sh test
!bash colab/run_quick_search.sh viz
```

## 10. Sinh Config Nhanh Kiểu SCRFD Gốc

Nếu bạn chỉ cần sinh vài config để test nhanh, không cần FLOPs filtering và không cần candidate search đầy đủ, dùng:

```bash
%cd /content/insightface/detection/scrfd
!bash colab/generate_quick_configs.sh
```

Mặc định script này:

- dùng template gốc `configs/scrfdgen2.5g/scrfdgen2.5g_0.py`
- sinh `4` config
- dùng search space kiểu SCRFD gốc
- không chạy FLOPs check

Ví dụ đổi sang mode 2 và sinh nhiều hơn:

```bash
%cd /content/insightface/detection/scrfd
!SCRFD_QUICK_GEN_MODE=2 SCRFD_QUICK_GEN_NUM_CONFIGS=8 \
  bash colab/generate_quick_configs.sh
```

Ví dụ đổi template và group đầu ra:

```bash
%cd /content/insightface/detection/scrfd
!SCRFD_QUICK_GEN_TEMPLATE=configs/scrfdgen2.5g/scrfdgen2.5g_0.py \
  SCRFD_QUICK_GEN_GROUP=configs/my_quick_test \
  SCRFD_QUICK_GEN_NUM_CONFIGS=3 \
  bash colab/generate_quick_configs.sh
```
