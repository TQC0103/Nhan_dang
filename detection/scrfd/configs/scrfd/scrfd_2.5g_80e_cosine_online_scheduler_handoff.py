_base_ = ['./scrfd_2.5g_80e_cosine_baseline.py']

redistribution_cfg = dict(
    STATE_KEY='scrfd_2_5g_online_scheduler_handoff_80e',
    ENABLE_ADAPTIVE_SR=True,
    ADAPTIVE_SR_WARMUP_EPOCHS=0,
    ADAPTIVE_SR_UPDATE_INTERVAL=0,
    ADAPTIVE_SR_EMA=0.8,
    ADAPTIVE_SR_MIN_PROB=0.03,
    ADAPTIVE_SR_SCALE_CANDIDATES=[0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
    ADAPTIVE_SR_BIN_EDGES=[0, 16, 32, 96, 100000000.0],
    ADAPTIVE_SR_DIFFICULTY_MODE='loss_recall',
    ADAPTIVE_SR_LOGGING=False,
    ENABLE_JSAR=False,
    JSAR_LOGGING=False,
)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),
    dict(
        type='RandomSquareCrop',
        crop_choice=redistribution_cfg['ADAPTIVE_SR_SCALE_CANDIDATES'],
        bbox_clip_border=False,
        adaptive_sr=redistribution_cfg,
    ),
    dict(
        type='Resize',
        img_scale=(640, 640),
        keep_ratio=False,
        bbox_clip_border=False),
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
            assigner=dict(type='ATSSAssigner', topk=9, mode=0),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
            redistribution_cfg=redistribution_cfg,
        )))

train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9, mode=0),
    allowed_border=-1,
    pos_weight=-1,
    debug=False,
    redistribution_cfg=redistribution_cfg,
)

custom_hooks = [
    dict(
        type='OnlineSchedulerHandoffHook',
        redistribution_cfg=redistribution_cfg,
        crop_choice=redistribution_cfg['ADAPTIVE_SR_SCALE_CANDIDATES'],
        state_file='online_scheduler_handoff_state.json',
        target_strides=(8, 16, 32),
        target_positive_ratios=(0.5, 0.3, 0.2),
        loss_weight=0.65,
        deficit_weight=0.35,
        update_momentum=0.6,
        temperature=0.8,
        min_crop_prob=0.03,
        priority='NORMAL',
    ),
]
