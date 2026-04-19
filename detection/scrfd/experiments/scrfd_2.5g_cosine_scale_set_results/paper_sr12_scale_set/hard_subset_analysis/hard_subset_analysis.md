# Hard Subset Analysis

## Aggregate

| Metric | Baseline | Improved | Delta |
| --- | ---: | ---: | ---: |
| hard_AP | 0.7371 | 0.7738 | +0.0367 |
| mAP | 0.8510 | 0.8556 | +0.0045 |
| hard_recall_proxy | 0.8357 | 0.8690 | +0.0333 |
| precision_proxy | 0.0167 | 0.0196 | +0.0029 |
| prediction_count | 1595279 | 1416440 | -178839 |

## Hard Recall By Size Bin

| Size Bin | GT | Baseline Recall Proxy | Improved Recall Proxy | Delta |
| --- | ---: | ---: | ---: | ---: |
| [0, 8) | 130 | 0.0615 | 0.1692 | +0.1077 |
| [8, 16) | 10627 | 0.6164 | 0.7203 | +0.1040 |
| [16, 32) | 10268 | 0.9086 | 0.9118 | +0.0032 |
| [32, 64) | 6650 | 0.9880 | 0.9767 | -0.0113 |
| [64, 128) | 2784 | 0.9917 | 0.9885 | -0.0032 |
| [128, inf) | 1499 | 0.9940 | 0.9907 | -0.0033 |
