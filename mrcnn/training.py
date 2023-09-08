import datetime
import os
import random
import uuid
import torch
from torch.utils.data import DataLoader, random_split
import optuna
from mrcnn import config
from mrcnn.dataset import CocoDataset
from mrcnn.model import MaskRCNNModel
from mrcnn.utilities import wrp, msg, save_hparam, pass_checkpoint, display_hparams, evaluation, load_checkpoint


@wrp
def start(root_directory, train_val_split=0.8, resume_from_checkpoint=None, batch_size=1,
          learning_rate=0.005, momentum=0.9, weight_decay=0.0005, lr_scheduler_factor=0.1, lr_scheduler_patience=3,
          num_epochs=20, score_thresh=0.05, backbone="resnet101",
          single_cat=False, opt="SGD"):
    # Creating dataset...
    dataset = CocoDataset(root_directory)

    dataset.run()
    root_path = dataset.images
    coco_json_path = dataset.annotations
    anchor_size = dataset.anchor_size
    anchor_ratio = dataset.anchor_ratio
    num_classes = dataset.num_classes

    # Splitting dataset into training and validation sets...

    train_size = int(train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size])

    # Creating data loaders...
    data_loader_train = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=lambda batch: tuple(zip(*batch)), shuffle=True,
        num_workers=6, pin_memory=True)
    data_loader_val = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=lambda batch: tuple(zip(*batch)), shuffle=True,
        num_workers=6, pin_memory=True)

    # Setting device...
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    MRCNN = MaskRCNNModel(num_classes, score_thresh,
                          backbone, anchor_size, anchor_ratio, device)

    params = [p for p in MRCNN.params if p.requires_grad]
    if opt.upper() == 'SGD':
        optimizer = torch.optim.SGD(
            params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    elif opt.upper() == 'ADAM':
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999),
                                     eps=1e-08)

    # Setting learning rate scheduler...
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_scheduler_factor, patience=lr_scheduler_patience, verbose=True)

    init_exiting_pat = 5
    exiting_pat = init_exiting_pat

    MRCNN.model, optimizer, start_epoch, best_map, best_train_loss, best_val_loss = load_checkpoint(root_path, backbone,
                                                                                                    MRCNN.model,
                                                                                                    optimizer, opt,
                                                                                                    num_classes)

    for epoch in range(start_epoch, num_epochs):
        op_id = uuid.uuid4()

        display_hparams(op_id, root_path, coco_json_path, device, backbone, batch_size,
                        learning_rate, momentum, weight_decay, lr_scheduler_factor,
                        lr_scheduler_patience, score_thresh)

        MRCNN.model, avg_train_loss, preds_formatted, gt_formatted, avg_train_loss_box_reg, avg_train_loss_classifier, avg_train_loss_objectness, avg_train_loss_rpn_box_reg = MRCNN.training(
            epoch, num_epochs, optimizer,
            data_loader_train, device, evaluate=False)

        MRCNN.model, avg_val_loss, preds_formatted, gt_formatted, avg_val_loss_box_reg, avg_val_loss_classifier, avg_val_loss_objectness, avg_val_loss_rpn_box_reg = MRCNN.validation(
            data_loader_val, epoch, device)

        metrics, total_map, mAP_50, mAP_75 = evaluation(op_id, dataset, single_cat, preds_formatted, gt_formatted)

        exiting_pat, best_map = pass_checkpoint(root_path, MRCNN.model, optimizer, epoch, best_train_loss,
                                                best_val_loss,
                                                best_map, total_map, backbone,
                                                init_exiting_pat, exiting_pat, num_epochs, momentum, device, batch_size,
                                                metrics,
                                                weight_decay, lr_scheduler_factor, op_id, coco_json_path,
                                                lr_scheduler_patience, score_thresh, learning_rate, anchor_size,
                                                anchor_ratio, mAP_50, mAP_75, avg_train_loss, avg_val_loss, opt,
                                                avg_train_loss_box_reg, avg_train_loss_classifier,
                                                avg_train_loss_objectness,
                                                avg_train_loss_rpn_box_reg, avg_val_loss_box_reg,
                                                avg_val_loss_classifier, avg_val_loss_objectness,
                                                avg_val_loss_rpn_box_reg)
        if not exiting_pat: return best_map

        lr_scheduler.step(avg_train_loss)  # Update the learning rate

    return best_map


def safe_start(root_path, backbone, single_cat, train_val_split, num_epochs, learning_rate, momentum, weight_decay,
               lr_scheduler_factor, lr_scheduler_patience, score_thresh):
    batch_size = config.batch_size
    split_size = config.split_size
    opt = config.opt
    best_map = 0

    while batch_size > 0 or split_size > 8:
        try:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{split_size}"
            best_map = start(
                root_path,
                batch_size=batch_size,
                train_val_split=train_val_split,
                learning_rate=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                lr_scheduler_factor=lr_scheduler_factor,
                lr_scheduler_patience=lr_scheduler_patience,
                num_epochs=num_epochs,
                score_thresh=score_thresh,
                backbone=backbone,
                single_cat=single_cat,
                opt=opt,
            )
        except Exception as e:
            print(e)
            if "CUDA out of memory" in str(e):

                msg(f"CUDA error likely, batch_size: {batch_size},  split_size: {split_size}, reducing...")
                batch_size //= 2
                split_size //= 2
                continue
            else:
                raise e
        break
    if batch_size < 1 or split_size < 8:
        raise Exception(f"YOUR {backbone} MODEL IS TOO BIG")
    return best_map


@wrp
def level(number, backbone, single_cat):
    time_started = datetime.datetime.now()

    root_path = config.dataset_root_path
    train_val_split = config.train_val_split
    num_epochs = config.num_epochs * number
    learning_rate = config.learning_rate
    momentum = config.momentum
    weight_decay = config.weight_decay
    lr_scheduler_factor = config.lr_scheduler_factor
    lr_scheduler_patience = config.lr_scheduler_patience
    score_thresh = config.score_thresh

    safe_start(
        root_path,
        train_val_split=train_val_split,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        num_epochs=num_epochs,
        score_thresh=score_thresh,
        backbone=backbone,
        single_cat=single_cat
    )

    time_ended = datetime.datetime.now()
    level_p = {"time_started": time_started, "time_ended": time_ended, "level": number, "backbone": backbone,
               "dataset": root_path,
               "single": single_cat}
    location = config.level_json
    save_hparam(hparams_file=location, hparam_dict=level_p)


def objective(trial):
    root_path = config.dataset_root_path

    # Suggest values for the parameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    momentum = trial.suggest_uniform('momentum', 0.7, 0.99)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    lr_scheduler_factor = trial.suggest_uniform('lr_scheduler_factor', 0.2, 0.9)
    lr_scheduler_patience = trial.suggest_int('lr_scheduler_patience', 3, 8)
    train_val_split = trial.suggest_uniform('train_val_split', 0.7, 0.8)
    score_thresh = 0.05
    backbone = trial.suggest_categorical('backbone', [config.backbones[0], ])
    num_epochs = 11

    try:
        curr_map = safe_start(
            root_path,
            train_val_split=train_val_split,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            num_epochs=num_epochs,
            score_thresh=score_thresh,
            backbone=backbone,
            single_cat=False,
        )

    except Exception as e:
        raise e
        print(e)
        msg("Something happend! continuing anyway...\n\n\n")
        curr_map = 0

    return curr_map  # Optuna maximizes this return value


@wrp
def optimize_hp(trials):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trials)
    # get best params
    best_params = study.best_params
    curr_map = study.best_value

    save_hparam(config.optimized_best_hp, {'curr_map': curr_map, 'best_params': best_params})

    # curr_map = 0
    # best_params = {
    #     "learning_rate": 4.464079455380031e-05,
    #     "momentum": 0.7009648808293939,
    #     "weight_decay": 0.0003026979044880114,
    #     "lr_scheduler_factor": 0.20030992565757083,
    #     "lr_scheduler_patience": 8,
    #     "train_val_split": 0.7438630634982245,
    #     "backbone": "resnet152"
    # }

    config.train_val_split = best_params['train_val_split']
    config.learning_rate = best_params['learning_rate']
    config.momentum = best_params['momentum']
    config.weight_decay = best_params['weight_decay']
    config.lr_scheduler_factor = best_params['lr_scheduler_factor']
    config.lr_scheduler_patience = best_params['lr_scheduler_patience']
    config.best_backbone = best_params['backbone']

    return curr_map


def main():
    curr_map = optimize_hp(trials=100)
    msg(f"Best mAP found:{curr_map}")

    backbone = config.best_backbone
    single_cat = False

    for iteration in range(1, 2 + 1):
        level(iteration, backbone, single_cat)


if __name__ == '__main__':
    main()
