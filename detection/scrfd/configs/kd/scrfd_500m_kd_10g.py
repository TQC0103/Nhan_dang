# Knowledge Distillation: SCRFD 10G -> SCRFD 500M

_base_ = './_base_kd_experiment.py'

model = dict(
    teacher=dict(
        type='SCRFD',
        pretrained=None,
        backbone=dict(
            type='ResNetV1e',
            depth=0,
            block_cfg=dict(
                block='BasicBlock',
                stage_blocks=(3, 4, 2, 3),
                stage_planes=[56, 88, 88, 224]
            ),
            base_channels=56,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            style='pytorch'
        ),
        neck=dict(
            type='PAFPN',
            in_channels=[56, 88, 88, 224],
            out_channels=56,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=3
        ),
        bbox_head=dict(
            type='SCRFDHead',
            num_classes=1,
            in_channels=56,
            stacked_convs=3,
            feat_channels=80,
            norm_cfg=dict(type='GN', num_groups=16, requires_grad=True),
            cls_reg_share=True,
            strides_share=True,
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
    ),
    distill_cfg=dict(
        cls_weight=0.5,
        bbox_weight=0.5,
        temperature=4.0,
        cls_loss_type='bce',
        match_by_stride=True,
        anchor_reduce='mean'
    )
)

total_epochs = 100
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.001,
    step=[55, 68]
)
