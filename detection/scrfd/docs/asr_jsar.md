# ASR + JSAR for SCRFD

This repository now includes a training-time extension for original SCRFD:

- `ASR`: Adaptive Sample Redistribution
- `JSAR`: Joint SampleAssignment Redistribution

The implementation is intentionally narrow. It does not change SCRFD backbone, neck, head structure, inference path, NAS flow, or add teacher-student distillation.

## Modified Files

- `mmdet/core/sample_redistribution.py`
  - shared runtime state
  - adaptive scale policy update
  - epoch JSON logging
  - `AdaptiveRedistributionHook`
- `mmdet/datasets/pipelines/transforms.py`
  - `RandomSquareCrop` can now read adaptive crop probabilities
- `mmdet/core/bbox/assigners/atss_assigner.py`
  - size-aware threshold relaxation
  - tiny/small-face fallback positive expansion
  - JSAR metadata for logging and weighting
- `mmdet/models/dense_heads/scrfd_head.py`
  - GT size histogram logging
  - per-level positive logging
  - per-size-bin cls/box loss summaries
  - JSAR before/after positive statistics
- `mmdet/apis/train.py`
  - imports the custom hook so it is registered for `custom_hooks`

## How ASR Works

`RandomSquareCrop` originally samples uniformly from a fixed `crop_choice` list. With ASR enabled:

1. The hook writes the current scale distribution to `work_dir/adaptive_sr/current_scale_probs.json`.
2. Data loader workers read that file and sample crop scale from the latest probabilities.
3. During training, SCRFDHead collects:
   - GT face histogram by size bin
   - matched positive anchors by size bin
   - per-level positive anchor counts
   - per-size-bin cls loss summary
   - per-size-bin box loss summary
4. Every `ADAPTIVE_SR_UPDATE_INTERVAL` iterations and at epoch end, the hook converts the current difficulty estimate into a new scale distribution with EMA smoothing and probability floor.

Difficulty is derived from a mix of:

- positive-per-GT recall proxy
- cls loss per positive anchor
- box loss per positive anchor

Larger zoom-in factors are biased toward tiny/small bins, while zoom-out factors are biased toward medium/large bins.

## How JSAR Works

JSAR is implemented on top of the existing `ATSSAssigner`.

When enabled:

- tiny/small GT boxes get a relaxed IoU threshold delta
- center gating is widened in a size-aware manner
- if a tiny/small GT still receives too few positives, the assigner force-adds a few nearby background anchors with the best overlap-distance score

Available modes:

- `size_aware_threshold`
  - only relax threshold and center gating
- `hybrid_fallback`
  - relax threshold/gating and then force-add positives when tiny/small GTs are still under-assigned
- `soft_weight`
  - same fallback path, but fallback positives also receive softer positive weights

This keeps medium/large faces close to original ATSS behavior while increasing supervision density for tiny/small faces.

## Enable / Disable

If `ENABLE_ADAPTIVE_SR=False` and `ENABLE_JSAR=False`, training follows the original SCRFD path.

Minimal common config block:

```python
redistribution_cfg = dict(
    STATE_KEY='scrfd_asr_jsar',
    ENABLE_ADAPTIVE_SR=False,
    ADAPTIVE_SR_WARMUP_EPOCHS=1,
    ADAPTIVE_SR_UPDATE_INTERVAL=1000,
    ADAPTIVE_SR_EMA=0.8,
    ADAPTIVE_SR_MIN_PROB=0.03,
    ADAPTIVE_SR_SCALE_CANDIDATES=[0.35, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
    ADAPTIVE_SR_BIN_EDGES=[0, 16, 32, 96, 1e8],
    ADAPTIVE_SR_DIFFICULTY_MODE='loss_recall',
    ADAPTIVE_SR_LOGGING=True,
    ENABLE_JSAR=False,
    JSAR_MODE='hybrid_fallback',
    JSAR_TINY_MAX_SIZE=16,
    JSAR_SMALL_MAX_SIZE=32,
    JSAR_TINY_IOU_DELTA=0.05,
    JSAR_SMALL_IOU_DELTA=0.02,
    JSAR_TOPK=4,
    JSAR_CENTER_RADIUS_SCALE=1.3,
    JSAR_SOFT_WEIGHT_TEMPERATURE=0.75,
    JSAR_MIN_POS_PER_TINY_GT=3,
    JSAR_LOGGING=True,
)
```

Baseline original SCRFD:

```python
train_pipeline = [
    dict(type='RandomSquareCrop',
         crop_choice=[0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]),
    ...
]

train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
)
```

SCRFD + ASR:

```python
train_pipeline = [
    dict(
        type='RandomSquareCrop',
        crop_choice=redistribution_cfg['ADAPTIVE_SR_SCALE_CANDIDATES'],
        adaptive_sr=redistribution_cfg,
    ),
    ...
]

train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    redistribution_cfg=dict(
        **redistribution_cfg,
        ENABLE_ADAPTIVE_SR=True,
        ENABLE_JSAR=False,
    ),
)

custom_hooks = [
    dict(
        type='AdaptiveRedistributionHook',
        redistribution_cfg=dict(
            **redistribution_cfg,
            ENABLE_ADAPTIVE_SR=True,
            ENABLE_JSAR=False,
        ),
        priority='NORMAL',
    ),
]
```

SCRFD + ASR + JSAR:

```python
train_pipeline = [
    dict(
        type='RandomSquareCrop',
        crop_choice=redistribution_cfg['ADAPTIVE_SR_SCALE_CANDIDATES'],
        adaptive_sr=dict(
            **redistribution_cfg,
            ENABLE_ADAPTIVE_SR=True,
            ENABLE_JSAR=True,
        ),
    ),
    ...
]

train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    redistribution_cfg=dict(
        **redistribution_cfg,
        ENABLE_ADAPTIVE_SR=True,
        ENABLE_JSAR=True,
        JSAR_MODE='hybrid_fallback',
    ),
)

custom_hooks = [
    dict(
        type='AdaptiveRedistributionHook',
        redistribution_cfg=dict(
            **redistribution_cfg,
            ENABLE_ADAPTIVE_SR=True,
            ENABLE_JSAR=True,
            JSAR_MODE='hybrid_fallback',
        ),
        priority='NORMAL',
    ),
]
```

## Debug Outputs

Per-run debug state is written under:

```text
<work_dir>/adaptive_sr/
```

Main files:

- `current_scale_probs.json`
- `latest_summary.json`
- `epoch_logs/epoch_XXX.json`

## Metrics to Watch

Focus first on tiny/small-face behavior:

- AP or recall for tiny/small faces if your evaluation split exposes it
- positive anchors per tiny/small GT
- `jsar_before_hist` vs `jsar_after_hist`
- `pos_hist / gt_hist` for tiny/small bins
- whether ASR shifts too much mass toward large zoom-in scales

## Expected Trade-offs / Failure Modes

- Too aggressive JSAR can add noisy positives and hurt precision.
- Too aggressive ASR can collapse scale diversity and overfit tiny faces.
- `soft_weight` is intentionally lightweight in this codebase: it softens fallback positives instead of redesigning the full dense target formulation.
- The runtime state is file-backed for compatibility with data loader workers, so the hook should be enabled whenever ASR is enabled.
