dataset_root_path = r"PATH"
level_json = dataset_root_path + r"/level.json"
optimized_best_hp = dataset_root_path + r"/optimized_best_hp.json"


backbones = [
    "resnet50",
    "resnet101",
    "resnet152",
]

best_backbone = None
batch_size = 64  # 2 if model == "resnet152" else 4 if model == "resnet101" else 8
split_size = 256
learning_rate = 1e-3  # 0.012
momentum = 0.78
weight_decay = 0.0001
lr_scheduler_factor = 0.9
lr_scheduler_patience = 8
score_thresh = 0.001
num_epochs = 300
train_val_split = 0.8
opt = "SGD"


