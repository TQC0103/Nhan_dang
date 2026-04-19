# SCRFD 2.5G Cosine Ablation Suite

This suite is intended to compare four training variants on WIDERFace with the
same SCRFD 2.5G searched architecture, all trained from scratch for 80 epochs:

- `baseline`
- `online_scheduler_handoff`
- `asr_jsar`
- `online_scheduler_handoff + asr_jsar`

The branch currently contains runnable cosine configs for:

- `configs/scrfd/scrfd_2.5g_80e_cosine_baseline.py`
- `configs/scrfd/scrfd_2.5g_80e_cosine_asr_jsar.py`
- `configs/scrfd/scrfd_2.5g_80e_cosine_online_scheduler_handoff.py`
- `configs/scrfd/scrfd_2.5g_80e_cosine_online_scheduler_handoff_asr_jsar.py`

The combined variant uses the online scheduler handoff for crop probability
updates and keeps `JSAR` enabled for assignment-side redistribution. It does
not run two crop schedulers at the same time.

## Run

```bash
cd detection/scrfd

bash colab/run_ablation_2.5g_cosine_suite.sh
```

Useful overrides:

```bash
BATCH_SIZE_PER_GPU=8
WORKERS_PER_GPU=2
BASELINE_GPU=0
OSH_GPU=1
ASR_JSAR_GPU=2
COMBO_GPU=3
LATENCY_WARMUP=30
LATENCY_REPEAT=200
```

## Outputs

Per-experiment result folders are written under:

```text
results/ablation_2.5g_cosine/<experiment>/
```

Important files:

- `results_summary.json`: WIDERFace AP summary
- `latency_summary.json`: mean/p50/p95 latency, FPS, peak memory, params, FLOPs

The suite comparison report is written to:

```text
results/ablation_2.5g_cosine/comparison/ablation_suite.md
results/ablation_2.5g_cosine/comparison/ablation_suite.csv
results/ablation_2.5g_cosine/comparison/ablation_suite.json
```

## Metrics In The Report

- `easy_AP`, `medium_AP`, `hard_AP`, `mAP`
- `latency_mean_ms`, `ms_per_image`, `latency_p50_ms`, `latency_p95_ms`
- `fps`
- `peak_memory_mb`
- `params_m`, `flops_g`
- `checkpoint_size_mb`
- `final_loss`, `final_loss_cls`, `final_loss_bbox`
- `avg_iter_time_s`, `train_hours`
