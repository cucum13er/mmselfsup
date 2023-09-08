# model settings
model = dict(
    type='MoCo_label',
    queue_len=8192,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='MoCoV2Neck',
        in_channels=512,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='SNNLossHead', temperature=0.07))
#    head=dict(type='ContrastiveHead', temperature=0.07))
