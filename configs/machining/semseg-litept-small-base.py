_base_ = ["../_base_/default_runtime.py"]

# Runtime. batch_size is the total batch size across all GPUs.
batch_size = 16
num_worker = 32
mix_prob = 0.0
empty_cache = False
enable_amp = True
clip_grad = 1.0
enable_wandb = False

# The child dataset configs replace these values.
save_path = "exp/machining/semseg-litept-small"

model = dict(
    type="DefaultSegmentorV2",
    num_classes=25,
    backbone_out_channels=72,
    backbone=dict(
        type="LitePT",
        # XYZ is carried separately as coord; the input feature is the normal.
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enc_mode=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# 100 optimizer epochs; each dataset is traversed once per epoch.
epoch = 100
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

dataset_type = "MachiningDataset"
data_root = ""
grid_size = 0.01

train_transform = [
    # The three source datasets use different physical scales. Normalize every
    # part to a unit sphere before voxelization so one grid size works for all.
    dict(type="NormalizeCoord"),
    dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomJitter", sigma=0.002, clip=0.01),
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": grid_size}),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment", "grid_size"),
        feat_keys=("normal",),
    ),
]

val_transform = [
    dict(type="NormalizeCoord"),
    dict(type="Copy", keys_dict={"segment": "origin_segment"}),
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        return_inverse=True,
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
        feat_keys=("normal",),
    ),
]

test_cfg = dict(
    voxelize=dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="test",
        return_grid_coord=True,
    ),
    crop=None,
    post_transform=[
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "index"),
            feat_keys=("normal",),
        ),
    ],
    # CAD orientation is meaningful; use one deterministic test view.
    aug_transform=[
        [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
    ],
)

data = dict(
    num_classes=25,
    ignore_index=-1,
    names=[f"class_{i}" for i in range(25)],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=train_transform,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=val_transform,
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[dict(type="NormalizeCoord")],
        test_mode=True,
        test_cfg=test_cfg,
    ),
)

