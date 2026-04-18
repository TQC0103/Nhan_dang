# SR Face-Size Distribution Analysis

This analysis uses the original `labelv2` face boxes and models the training-time size change as:

`augmented_face_size ~= original_face_size * resize_size / (crop_scale * min(image_w, image_h))`

It does not simulate image content or face dropping from crop position. The goal is to compare the *size distribution pressure* induced by each sampling policy.

## Inputs

- Annotation file: `E:\GHuy\Study\Nhận dạng\SCRFD\analysis_artifacts_bundle\dataset_metadata\train\labelv2.txt`
- Baseline config: `E:\GHuy\Study\Nhận dạng\SCRFD\insightface\detection\scrfd\configs\scrfd\scrfd_2.5g_80e_cosine_baseline_paper_sr12.py`
- Improved config: `E:\GHuy\Study\Nhận dạng\SCRFD\insightface\detection\scrfd\configs\scrfd\scrfd_2.5g_80e_cosine_asr_jsar_paper_sr12.py`
- Improved scale source: `{'source': 'train_log_mean', 'resolved_source': 'E:\\tmp\\ab2\\experiments\\asr_jsar\\work_dir\\adaptive_sr\\scale_prob_history.jsonl', 'num_records': 78, 'warmup_epochs_skipped': 2}`

## Key Numbers

| Distribution | mean | p50 | p90 | tiny | small | medium | large | tiny->>=16 | <32->>=32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 32.76 | 17.44 | 65.73 | 0.4622 | 0.2737 | 0.2055 | 0.0586 | - | - |
| Baseline SR | 25.12 | 12.97 | 53.34 | 0.5816 | 0.2191 | 0.1594 | 0.0399 | 0.0920 | 0.0464 |
| ASR+JSAR | 25.06 | 13.18 | 52.91 | 0.5770 | 0.2237 | 0.1603 | 0.0390 | 0.0808 | 0.0399 |

## Interpretation Hints

- If `ASR+JSAR` moves more mass out of the `tiny` bin and into `small` / `medium`, that supports the claim that tiny faces receive stronger supervision.
- `tiny->>=16` is a direct proxy for how often originally tiny faces become at least small after augmentation.
- `<32->>=32` is a stronger promotion proxy: small hard faces becoming medium-sized for training.
- The transition heatmaps show whether the improved policy concentrates more probability on the `tiny -> small` and `tiny -> medium` routes.
