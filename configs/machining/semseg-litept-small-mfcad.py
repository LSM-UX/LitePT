_base_ = ["./semseg-litept-small-base.py"]

data_root = "/data/xuzg/LSM/PointNeXt-master/traindata/MFCAD"
save_path = "exp/machining/mfcad/semseg-litept-small"

model = dict(num_classes=16)
data = dict(
    num_classes=16,
    names=[f"class_{i}" for i in range(16)],
    train=dict(data_root=data_root),
    val=dict(data_root=data_root),
    test=dict(data_root=data_root),
)

