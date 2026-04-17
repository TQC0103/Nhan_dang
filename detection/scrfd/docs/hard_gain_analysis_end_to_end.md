# Hard Gain Analysis End-To-End

This guide covers the full workflow for:

1. training `SCRFD-2.5G baseline` and `SCRFD-2.5G ASR+JSAR`
2. evaluating both on WIDERFace
3. plotting scale-probability history from train logs
4. analyzing face-size distribution shifts caused by SR / ASR
5. building one bundled report that explains why `hard_AP` improved
6. collecting logs and artifacts in one place for download

All commands below assume you are inside:

```bash
cd insightface/detection/scrfd
```

## 1. Environment

If you are on VPS and already ran `setup_vps_env.sh`, activate:

```bash
export MAMBA_ROOT_PREFIX="$HOME/.local/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook -s bash)"
micromamba activate scrfd-vps
```

Check quickly:

```bash
python -c "import torch, mmcv, mmdet; print(torch.__version__, mmcv.__version__, mmdet.__version__)"
```

## 2. Prepare Dataset

If `data/retinaface/train/labelv2.txt` and `data/retinaface/val/gt` are not ready yet:

```bash
bash prepare_retinaface_data.sh --download-all --force
```

Check:

```bash
ls data/retinaface/train/labelv2.txt
ls data/retinaface/val/labelv2.txt
ls data/retinaface/val/gt | head
```

## 3. Train Baseline And ASR+JSAR

Manual training:

```bash
python tools/train.py \
  configs/scrfd/scrfd_2.5g_80e_baseline.py \
  --work-dir work_dirs/compare_2.5g/baseline

python tools/train.py \
  configs/scrfd/scrfd_2.5g_80e_asr_jsar.py \
  --work-dir work_dirs/compare_2.5g/asr_jsar
```

If you use the compare runner:

```bash
bash colab/run_compare_2.5g_baseline_vs_asr_jsar.sh
```

Important:

- New runs now write scale-probability history to:

```text
work_dirs/.../adaptive_sr/scale_prob_history.jsonl
```

- That file is the preferred source for all later plots and reports.

## 4. Evaluate Both Models

To enable hard-subset analysis later, evaluate with `--save-preds`.

```bash
python tools/test_widerface_enhanced.py \
  configs/scrfd/scrfd_2.5g_80e_baseline.py \
  work_dirs/compare_2.5g/baseline/latest.pth \
  --out results/compare_2.5g/baseline \
  --save-preds

python tools/test_widerface_enhanced.py \
  configs/scrfd/scrfd_2.5g_80e_asr_jsar.py \
  work_dirs/compare_2.5g/asr_jsar/latest.pth \
  --out results/compare_2.5g/asr_jsar \
  --save-preds
```

You should then have:

```text
results/compare_2.5g/baseline/results_summary.json
results/compare_2.5g/baseline/predictions/
results/compare_2.5g/asr_jsar/results_summary.json
results/compare_2.5g/asr_jsar/predictions/
```

## 5. Compare WIDERFace Metrics

```bash
python tools/compare_widerface_results.py \
  --baseline results/compare_2.5g/baseline \
  --improved results/compare_2.5g/asr_jsar \
  --baseline-name "SCRFD-2.5G Baseline 80e" \
  --improved-name "SCRFD-2.5G ASR+JSAR 80e" \
  --out-dir results/compare_2.5g/comparison
```

Main output:

```text
results/compare_2.5g/comparison/comparison.md
```

## 6. Plot Scale Probability History

This plots one line per scale candidate using the train-time history.

```bash
bash tools/run_plot_scale_prob_history.sh \
  work_dirs/compare_2.5g/asr_jsar \
  results/compare_2.5g/asr_jsar_scale_prob_history \
  configs/scrfd/scrfd_2.5g_80e_asr_jsar.py
```

Main outputs:

```text
results/compare_2.5g/asr_jsar_scale_prob_history/scale_prob_history_epoch_end.png
results/compare_2.5g/asr_jsar_scale_prob_history/scale_prob_history.csv
results/compare_2.5g/asr_jsar_scale_prob_history/scale_prob_history_summary.json
```

Notes:

- For new runs, the script reads `scale_prob_history.jsonl`.
- For old runs, it can fall back to parsing the text `.log`.

## 7. Analyze Face-Size Distribution

This compares:

- original dataset face-size distribution
- baseline SR-induced distribution
- ASR+JSAR distribution using the actual train-time scale probabilities

```bash
bash tools/run_sr_face_size_distribution_analysis.sh \
  configs/scrfd/scrfd_2.5g_80e_baseline.py \
  configs/scrfd/scrfd_2.5g_80e_asr_jsar.py \
  work_dirs/compare_2.5g/asr_jsar \
  results/compare_2.5g/face_size_analysis \
  data/retinaface/train/labelv2.txt
```

Main outputs:

```text
results/compare_2.5g/face_size_analysis/face_size_histogram.png
results/compare_2.5g/face_size_analysis/face_size_cdf.png
results/compare_2.5g/face_size_analysis/face_size_bin_ratios.png
results/compare_2.5g/face_size_analysis/scale_probabilities.png
results/compare_2.5g/face_size_analysis/baseline_transition_heatmap.png
results/compare_2.5g/face_size_analysis/asr_jsar_transition_heatmap.png
results/compare_2.5g/face_size_analysis/analysis.md
```

The most useful fields in the summary:

- `tiny->>=16`
- `<32->>=32`
- `tiny/small/medium/large` ratios

These are the clearest signals for why `hard_AP` moved.

## 8. Hard Subset Analysis

If you evaluated with `--save-preds`, you can analyze hard faces directly:

```bash
python tools/analyze_hard_subset_comparison.py \
  --baseline results/compare_2.5g/baseline \
  --improved results/compare_2.5g/asr_jsar \
  --gt-dir data/retinaface/val/gt \
  --out-dir results/compare_2.5g/hard_analysis
```

Main outputs:

```text
results/compare_2.5g/hard_analysis/hard_subset_analysis.md
results/compare_2.5g/hard_analysis/hard_size_bins.csv
results/compare_2.5g/hard_analysis/top_improved_images.csv
```

If `predictions/` is missing, rerun evaluation with `--save-preds`.

## 9. Build One Bundled Explanation Report

This step runs the comparison pieces again, builds one markdown report, and gathers logs into one place.

```bash
bash tools/run_build_hard_gain_explanation_report.sh \
  results/compare_2.5g/baseline \
  results/compare_2.5g/asr_jsar \
  work_dirs/compare_2.5g/baseline \
  work_dirs/compare_2.5g/asr_jsar \
  results/compare_2.5g/hard_gain_report \
  data/retinaface/val/gt \
  data/retinaface/train/labelv2.txt
```

Main outputs:

```text
results/compare_2.5g/hard_gain_report/report.md
results/compare_2.5g/hard_gain_report/report_data.json
results/compare_2.5g/hard_gain_report/logs/
results/compare_2.5g/hard_gain_report/logs_bundle.zip
results/compare_2.5g/hard_gain_report/report_bundle.zip
```

## 10. Where Logs Are Kept

For each new training run:

```text
work_dirs/.../*.log
work_dirs/.../*.log.json
work_dirs/.../adaptive_sr/current_scale_probs.json
work_dirs/.../adaptive_sr/scale_prob_history.jsonl
work_dirs/.../adaptive_sr/latest_summary.json
work_dirs/.../adaptive_sr/epoch_logs/
```

For the final bundled report:

```text
results/.../hard_gain_report/logs/
results/.../hard_gain_report/logs_bundle.zip
```

`hard_gain_report/logs/` is the easiest place to download from, because it collects:

- train logs
- eval logs
- adaptive SR / handoff logs
- generated analysis files
- step-by-step report build logs
- `manifest.json` showing what was copied from where

## 11. Recommended Minimal End-To-End Command Sequence

If you already trained:

```bash
python tools/test_widerface_enhanced.py \
  configs/scrfd/scrfd_2.5g_80e_baseline.py \
  work_dirs/compare_2.5g/baseline/latest.pth \
  --out results/compare_2.5g/baseline \
  --save-preds

python tools/test_widerface_enhanced.py \
  configs/scrfd/scrfd_2.5g_80e_asr_jsar.py \
  work_dirs/compare_2.5g/asr_jsar/latest.pth \
  --out results/compare_2.5g/asr_jsar \
  --save-preds

bash tools/run_build_hard_gain_explanation_report.sh \
  results/compare_2.5g/baseline \
  results/compare_2.5g/asr_jsar \
  work_dirs/compare_2.5g/baseline \
  work_dirs/compare_2.5g/asr_jsar \
  results/compare_2.5g/hard_gain_report \
  data/retinaface/val/gt \
  data/retinaface/train/labelv2.txt
```

If you want the intermediate plots separately too:

```bash
bash tools/run_plot_scale_prob_history.sh \
  work_dirs/compare_2.5g/asr_jsar \
  results/compare_2.5g/asr_jsar_scale_prob_history \
  configs/scrfd/scrfd_2.5g_80e_asr_jsar.py

bash tools/run_sr_face_size_distribution_analysis.sh \
  configs/scrfd/scrfd_2.5g_80e_baseline.py \
  configs/scrfd/scrfd_2.5g_80e_asr_jsar.py \
  work_dirs/compare_2.5g/asr_jsar \
  results/compare_2.5g/face_size_analysis \
  data/retinaface/train/labelv2.txt
```

## 12. What To Cite In Your Explanation

When writing the explanation for why `ASR+JSAR` improves `hard_AP`, the strongest chain is:

1. `scale_prob_history_epoch_end.png`
   Shows the policy shifted toward scales that better expose tiny/small faces.
2. `face_size_bin_ratios.png` and the transition heatmaps
   Shows more mass moved from `tiny` toward `small/medium`.
3. `hard_subset_analysis.md`
   Shows recall proxy improved specifically on hard faces and on small size bins.
4. `latest_summary.json` under `adaptive_sr`
   Shows `jsar_before_hist` vs `jsar_after_hist`, proving extra positive assignment for tiny/small bins.

That combination is usually enough to justify the hard-face gain without needing to inspect individual images first.
