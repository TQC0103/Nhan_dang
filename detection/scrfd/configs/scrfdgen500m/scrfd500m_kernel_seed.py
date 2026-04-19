# Kernel-NAS seed built from the searched SCRFD 500M architecture.
# This keeps the searched 500M width/depth/head design and only adds kernel
# fields so further NAS can focus on stem/stage kernels.

_base_ = [
    '../scrfd/scrfd_500m.py'
]

model = dict(
    type='SCRFD',
    backbone=dict(
        type='MobileNetV1KS',
        block_cfg=dict(
            stage_blocks=(2, 3, 2, 6),
            stage_planes=[16, 16, 40, 72, 152, 288],
            stem_kernel_size=3,
            stem_dw_kernel_size=3,
            stage_kernel_sizes=[3, 3, 3, 3],
        ),
        out_indices=(0, 1, 2, 3),
    ),
    neck=dict(
        type='PAFPN',
        in_channels=[40, 72, 152, 288],
        out_channels=16,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=3,
    ),
    bbox_head=dict(
        type='SCRFDHead',
        num_classes=1,
        in_channels=16,
        stacked_convs=2,
        feat_channels=64,
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
            strides=[8, 16, 32],
        ),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0,
        ),
        loss_dfl=False,
        reg_max=8,
        loss_bbox=dict(type='DIoULoss', loss_weight=2.0),
        use_kps=False,
    ),
)

# Keep search practical: inherit the searched 500M architecture, but use
# shorter search-time training defaults instead of the full 640-epoch schedule.
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.001,
    step=[55, 68],
)
total_epochs = 80
checkpoint_config = dict(interval=80)
evaluation = dict(interval=80, metric='mAP')
