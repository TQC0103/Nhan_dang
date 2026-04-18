# Hard Subset Analysis

## Aggregate

| Metric | Baseline | Improved | Delta |
| --- | ---: | ---: | ---: |
| hard_AP | 0.7073 | 0.7457 | +0.0384 |
| mAP | 0.8310 | 0.8341 | +0.0030 |
| hard_recall_proxy | 0.8175 | 0.5805 | -0.2369 |
| precision_proxy | 0.0134 | 0.0171 | +0.0037 |
| prediction_count | 1949484 | 1081972 | -867512 |

## Hard Recall By Size Bin

| Size Bin | GT | Baseline Recall Proxy | Improved Recall Proxy | Delta |
| --- | ---: | ---: | ---: | ---: |
| [0, 8) | 130 | 0.0462 | 0.0846 | +0.0385 |
| [8, 16) | 10627 | 0.5731 | 0.5124 | -0.0607 |
| [16, 32) | 10268 | 0.9004 | 0.6325 | -0.2678 |
| [32, 64) | 6650 | 0.9836 | 0.6308 | -0.3528 |
| [64, 128) | 2784 | 0.9878 | 0.5740 | -0.4138 |
| [128, inf) | 1499 | 0.9960 | 0.5397 | -0.4563 |
