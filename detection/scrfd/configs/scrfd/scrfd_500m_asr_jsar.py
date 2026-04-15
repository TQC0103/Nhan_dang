_base_ = ['./scrfd_500m.py']

redistribution_cfg = dict(
    STATE_KEY='scrfd_500m_asr_jsar',
    ENABLE_ADAPTIVE_SR=True,
    ADAPTIVE_SR_WARMUP_EPOCHS=2,
    ADAPTIVE_SR_UPDATE_INTERVAL=1000,
    ADAPTIVE_SR_EMA=0.8,
    ADAPTIVE_SR_MIN_PROB=0.03,
    ADAPTIVE_SR_SCALE_CANDIDATES=[0.35, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
    ADAPTIVE_SR_BIN_EDGES=[0, 16, 32, 96, 100000000.0],
    ADAPTIVE_SR_DIFFICULTY_MODE='loss_recall',
    ADAPTIVE_SR_LOGGING=True,
    ENABLE_JSAR=True,
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

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),
    dict(
        type='RandomSquareCrop',
        crop_choice=redistribution_cfg['ADAPTIVE_SR_SCALE_CANDIDATES'],
        adaptive_sr=redistribution_cfg,
    ),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[128.0, 128.0, 128.0],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_keypointss'])
]

data = dict(train=dict(pipeline=train_pipeline))

model = dict(
    bbox_head=dict(
        train_cfg=dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
            redistribution_cfg=redistribution_cfg,
        )))

train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False,
    redistribution_cfg=redistribution_cfg,
)

custom_hooks = [
    dict(
        type='AdaptiveRedistributionHook',
        redistribution_cfg=redistribution_cfg,
        priority='NORMAL',
    ),
]
