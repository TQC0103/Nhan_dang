# Hard Subset Analysis

## Aggregate

| Metric | Baseline | Improved | Delta |
| --- | ---: | ---: | ---: |
| hard_AP | 0.7249 | 0.7690 | +0.0440 |
| mAP | 0.8453 | 0.8551 | +0.0099 |
| hard_recall_proxy | 0.8243 | 0.8659 | +0.0416 |
| precision_proxy | 0.0153 | 0.0182 | +0.0029 |
| prediction_count | 1725098 | 1522654 | -202444 |

## Hard Recall By Size Bin

| Size Bin | GT | Baseline Recall Proxy | Improved Recall Proxy | Delta |
| --- | ---: | ---: | ---: | ---: |
| [0, 8) | 130 | 0.0231 | 0.1615 | +0.1385 |
| [8, 16) | 10627 | 0.5917 | 0.7107 | +0.1190 |
| [16, 32) | 10268 | 0.9035 | 0.9123 | +0.0089 |
| [32, 64) | 6650 | 0.9823 | 0.9755 | -0.0068 |
| [64, 128) | 2784 | 0.9896 | 0.9896 | +0.0000 |
| [128, inf) | 1499 | 0.9933 | 0.9927 | -0.0007 |
