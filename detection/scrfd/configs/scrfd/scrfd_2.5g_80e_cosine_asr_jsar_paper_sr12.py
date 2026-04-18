_base_ = ['./scrfd_2.5g_80e_asr_jsar_paper_sr12.py']

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.001,
    min_lr=0.0,
)
