# Template config for SCRFD 500M kernel search
# This config is used as a starting point for kernel size architecture search

_base_ = [
    '../scrfd/scrfd_500m.py'
]

# Model with kernel-search capable backbone
model = dict(
    type='SCRFD',
    backbone=dict(
        type='MobileNetV1KS',  # Kernel-search capable backbone
        block_cfg=dict(
            stage_blocks=(2, 2, 6, 3),
            stage_planes=[8, 16, 32, 64, 128, 256],
            stem_kernel_size=3,  # The very first conv is now part of NAS
            stem_dw_kernel_size=3,
            stage_kernel_sizes=[3, 3, 3, 3]  # Default 3x3, will be searched
        ),
        num_stages=4,
        out_indices=(0, 1, 2, 3)
    ),
    neck=dict(
        type='PAFPN',
        in_channels=[32, 64, 128, 256],
        out_channels=32,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=3
    ),
    bbox_head=dict(
        type='SCRFDHead',
        num_classes=1,
        in_channels=32,
        stacked_convs=2,
        feat_channels=80,
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True),
        cls_reg_share=True,
        strides_share=True,
        dw_conv=True,
        scale_mode=2,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[1, 2],
            base_sizes=[16, 64, 256],
            strides=[8, 16, 32]
        ),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0
        ),
        loss_dfl=False,
        reg_max=8,
        loss_bbox=dict(type='DIoULoss', loss_weight=2.0),
        use_kps=False
    )
)

# Training settings
train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False
)

test_cfg = dict(
    nms_pre=-1,
    min_bbox_size=0,
    score_thr=0.02,
    nms=dict(type='nms', iou_threshold=0.45),
    max_per_img=-1
)

# Training hyperparameters
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.001,
    step=[55, 68]
)
total_epochs = 80
checkpoint_config = dict(interval=80)
evaluation = dict(interval=80, metric='mAP')

# Expected FLOPs: ~0.5G (500M)
