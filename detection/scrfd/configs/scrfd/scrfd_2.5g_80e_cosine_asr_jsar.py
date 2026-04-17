_base_ = ['./scrfd_2.5g_80e_asr_jsar.py']

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.001,
    min_lr=0.0,
)
