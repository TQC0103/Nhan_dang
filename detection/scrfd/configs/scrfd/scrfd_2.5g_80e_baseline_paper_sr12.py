_base_ = ['./scrfd_2.5g_80e_baseline.py']

paper_sr12_scale_candidates = [0.5, 0.7, 0.8, 1.0, 1.1, 1.2, 1.4, 1.5, 1.8, 2.0, 2.3, 2.6]

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),
    dict(
        type='RandomSquareCrop',
        crop_choice=paper_sr12_scale_candidates,
        bbox_clip_border=False),
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
            redistribution_cfg=dict(
                STATE_KEY='scrfd_2_5g_baseline_paper_sr12',
                ENABLE_ADAPTIVE_SR=False,
                ENABLE_JSAR=False,
            ),
        )))

train_cfg = dict(
    redistribution_cfg=dict(
        STATE_KEY='scrfd_2_5g_baseline_paper_sr12',
        ENABLE_ADAPTIVE_SR=False,
        ENABLE_JSAR=False,
    ),
)
