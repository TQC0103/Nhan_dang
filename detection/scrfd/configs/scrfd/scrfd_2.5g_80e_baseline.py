_base_ = ['./scrfd_2.5g.py']

total_epochs = 80

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.001,
    step=[55, 68],
)

checkpoint_config = dict(interval=80)
evaluation = dict(interval=80, metric='mAP')

model = dict(
    bbox_head=dict(
        train_cfg=dict(
            assigner=dict(type='ATSSAssigner', topk=9, mode=0),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
            redistribution_cfg=dict(
                STATE_KEY='scrfd_2_5g_baseline',
                ENABLE_ADAPTIVE_SR=False,
                ENABLE_JSAR=False,
            ),
        )))

train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9, mode=0),
    allowed_border=-1,
    pos_weight=-1,
    debug=False,
    redistribution_cfg=dict(
        STATE_KEY='scrfd_2_5g_baseline',
        ENABLE_ADAPTIVE_SR=False,
        ENABLE_JSAR=False,
    ),
)
