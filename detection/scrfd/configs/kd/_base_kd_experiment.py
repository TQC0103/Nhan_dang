# Base configuration for Knowledge Distillation experiments.

_base_ = [
    '../scrfd/scrfd_500m.py'
]

custom_imports = dict(
    imports=['mmdet.core.distillation'],
    allow_failed_imports=False)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

custom_hooks = [
    dict(
        type='KDTensorboardLoggerHook',
        interval=10,
        csv_log_file='${work_dir}/loss_log.csv')
]

model = dict(
    type='SCRFDKD',
    teacher=dict(
        type='SCRFD',
        pretrained=None,
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

checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='mAP')
