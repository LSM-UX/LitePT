_base_ = ["./semseg-litept-small-base.py"]

data_root = "/data/xuzg/LSM/PointNeXt-master/traindata/MFCAD2"
save_path = "exp/machining/mfcad2/semseg-litept-small"

model = dict(num_classes=25)
data = dict(
    num_classes=25,
    names=[f"class_{i}" for i in range(25)],
    train=dict(data_root=data_root),
    val=dict(data_root=data_root),
    test=dict(data_root=data_root),
)

