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

## 3. Setup môi trường

```bash
%cd /content/insightface
!bash detection/scrfd/colab/setup_colab_env.sh /content/insightface
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
