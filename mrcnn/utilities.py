import datetime
import json
import os
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pycocotools import mask as maskUtils


def msg(msg, indent=1, width=None, title=None):
    """Print message-box with optional title."""
    lines = msg.split('\n')
    space = " " * indent
    if not width:
        width = max(map(len, lines))
    box = f'\n╔{"═" * (width + indent * 2)}╗\n'  # upper_border
    if title:
        box += f'║{space}{title:<{width}}{space}║\n'  # title
        box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
    box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
    box += f'╚{"═" * (width + indent * 2)}╝\n'  # lower_border
    print(box)


def wrp(func):
    def inner(*args, **kwargs):
        __border = "=" * 50 + "=" * len(func.__name__) + "=" * 6 + "=" * 50
        print()
        print(__border)
        print("=" * 50 + f" [ {func.__name__} ] " + "=" * 50)
        print(__border)
        argstr = str(args)
        kwrgstr = str(kwargs)
        lmt = 500
        if len(argstr) < lmt:
            print(argstr)
        if len(kwrgstr) < lmt:
            print(kwrgstr)
        print(__border)
        results = func(*args, **kwargs)
        print(__border)
        print()
        return results

    return inner


@wrp
def save_hparam(hparams_file, hparam_dict):
    msg(f"Saving: {hparams_file}")
    Path(str(os.path.dirname(hparams_file))).mkdir(parents=True, exist_ok=True)
    prev_best = []
    if os.path.isfile(hparams_file):
        with open(hparams_file, 'r') as f:
            prev_best = json.load(f)
    prev_best.append(hparam_dict)
    with open(hparams_file, 'w') as f:
        json.dump(prev_best, f, default=str, indent=4, separators=(',', ': '))


def get_hparam_dict(op_id, root_path, coco_json_path, device, model_name, batch_size, learning_rate, momentum,
                    weight_decay, lr_scheduler_factor, lr_scheduler_patience, score_thresh, num_epochs=None,
                    best_map=None, epoch=None,
                    anchor_size=None, anchor_ratio=None, mAP_50=None, mAP_75=None, avg_train_loss=None,
                    avg_val_loss=None, full=True, opt=None, avg_train_loss_box_reg=None, avg_train_loss_classifier=None,
                    avg_train_loss_objectness=None, avg_train_loss_rpn_box_reg=None, avg_val_loss_box_reg=None,
                    avg_val_loss_classifier=None, avg_val_loss_objectness=None, avg_val_loss_rpn_box_reg=None):
    hparams_dict = {
        "op_id": op_id,
        "coco_dataset": coco_json_path,
        "device": device,
        "model_name": model_name,
        "optimizer": opt,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "lr_scheduler_factor": lr_scheduler_factor,
        "lr_scheduler_patience": lr_scheduler_patience,
        "score_thresh": score_thresh,
        "num_epochs": num_epochs,
        "best_map": best_map,
        "epoch": epoch,
        "anchor_size": anchor_size,
        "anchor_ratio": anchor_ratio,
        "time_stamp": datetime.datetime.now(),
        "mAP_50": mAP_50,
        "mAP_75": mAP_75,
        "training_loss": avg_train_loss,
        "avg_train_loss_box_reg": avg_train_loss_box_reg,
        "avg_train_loss_classifier": avg_train_loss_classifier,
        "avg_train_loss_objectness": avg_train_loss_objectness,
        "avg_train_loss_rpn_box_reg": avg_train_loss_rpn_box_reg,
        "validation_loss": avg_val_loss,
        "avg_val_loss_box_reg": avg_val_loss_box_reg,
        "avg_val_loss_classifier": avg_val_loss_classifier,
        "avg_val_loss_objectness": avg_val_loss_objectness,
        "avg_val_loss_rpn_box_reg": avg_val_loss_rpn_box_reg,
    } if full else {
        "op_id": op_id,
        "coco_dataset": coco_json_path,
        "device": device,
        "model_name": model_name,
        "optimizer": opt,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "lr_scheduler_factor": lr_scheduler_factor,
        "lr_scheduler_patience": lr_scheduler_patience,
        "score_thresh": score_thresh,
        "num_epochs": num_epochs,
        "anchor_size": anchor_size,
        "anchor_ratio": anchor_ratio,
        "time_stamp": datetime.datetime.now(),
    }
    hparams_file = f'{root_path}/progress/hparams.json'
    return hparams_dict, hparams_file


@wrp
def display_hparams(op_id=None, root_path=None, coco_json_path=None, device=None, model_name=None, batch_size=None,
                    learning_rate=None, momentum=None, weight_decay=None, lr_scheduler_factor=None,
                    lr_scheduler_patience=None, score_thresh=None):
    hparam_dict, _ = get_hparam_dict(op_id, root_path, coco_json_path, device, model_name, batch_size, learning_rate,
                                     momentum,
                                     weight_decay, lr_scheduler_factor, lr_scheduler_patience, score_thresh, full=False)
    pprint(hparam_dict)


@wrp
def load_checkpoint(root_path, model_name, model, optimizer, opt, num_classes):
    best_model_path = f'{root_path}/progress/fasterrcnn_{model_name}_fpn_{opt}_best_model.pth'
    best_model_path_exist = os.path.exists(best_model_path)

    latest_model_path = f'{root_path}/progress/fasterrcnn_{model_name}_fpn_{opt}_loss.pth'
    latest_model_path_exist = os.path.exists(latest_model_path)

    resume_from_checkpoint = None
    if best_model_path_exist:
        if latest_model_path_exist:
            resume_from_checkpoint = latest_model_path if os.path.getctime(best_model_path) < os.path.getctime(
                latest_model_path) else best_model_path
        else:
            resume_from_checkpoint = best_model_path

        resume_from_checkpoint = best_model_path  # OVERRIDE
    elif latest_model_path_exist:
        resume_from_checkpoint = latest_model_path

    if resume_from_checkpoint is not None:
        msg("Loading checkpoint...")
        try:
            checkpoint = torch.load(resume_from_checkpoint)
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                print(e)
                # Load state dict, but ignore the final layer (the classifier) as it has a different size
                model.load_state_dict(
                    {k: v for k, v in checkpoint['model_state_dict'].items() if 'box_predictor' not in k}, strict=False)

                # Replace the box predictor with a new one
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

                # Replace the mask predictor
                in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
                hidden_layer = 256
                model.roi_heads.mask_predictor = MaskRCNNPredictor(
                    in_features_mask, hidden_layer, num_classes)

            # Copy over the weights from the pretrained model, except for the RPN
            # for name, param in checkpoint['model_state_dict'].items():
            #     if 'rpn' not in name:
            #         model.state_dict()[name].copy_(param)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_map = checkpoint.get('best_map', 0)
            best_train_loss = checkpoint.get('best_train_loss', 0)
            best_val_loss = checkpoint.get('best_val_loss', 0)
            if best_val_loss == -1:
                best_val_loss = best_train_loss * 1.5  # added 50%
        except Exception as e:
            raise Exception(f"Failed to load checkpoint. Error: {e}")
        return model, optimizer, start_epoch, best_map, best_train_loss, best_val_loss
    return model, optimizer, 0, 0, 0, 0


def old_format_predictions(annotations, preds, preds_formatted=None, gt_formatted=None):
    # Format the predictions and ground-truths
    id_counter = 0  # Initialize a counter for unique annotation ids
    for i, pred in enumerate(preds):
        for j in range(len(pred['boxes'])):
            rle = maskUtils.encode(np.asfortranarray(pred['masks'][j].cpu().numpy().astype(np.uint8)))

            preds_formatted.append({'image_id': i,
                                    'category_id': pred['labels'][j].item(),
                                    'bbox': pred['boxes'][j].tolist(),
                                    'segmentation': pred['masks'][j].tolist(),  # Include masks
                                    'score': pred['scores'][j].item()})

        for j in range(len(annotations[i]['boxes'])):
            rle = maskUtils.encode(np.asfortranarray(annotations[i]['masks'][j].cpu().numpy().astype(np.uint8)))

            gt_formatted.append({'image_id': i, 'category_id': annotations[i]['labels'][j].item(),
                                 'bbox': annotations[i]['boxes'][j].tolist(),
                                 'area': annotations[i]['area'][j].item(),
                                 'iscrowd': annotations[i]['iscrowd'][j].item(),
                                 'segmentation': pred['masks'][j].tolist(),  # Include masks
                                 'id': id_counter})  # Add the unique id to each annotation
            id_counter += 1  # Increment the counter for each annotation

    return preds_formatted, gt_formatted


def get_loss_dict(model, imgs, annotations):
    model.train()
    ld = model(imgs, annotations)
    model.eval()
    return ld


def get_preds(model, imgs):
    model.eval()
    preds = model(imgs)
    model.train()
    return preds


def get_losses(loss_dict):
    loss_box_reg, loss_classifier, loss_objectness, loss_rpn_box_reg = 9999, 9999, 9999, 9999

    for k, v in loss_dict.items():
        match k:
            case "loss_box_reg":
                loss_box_reg = float(v)
            case "loss_classifier":
                loss_classifier = float(v)
            case "loss_objectness":
                loss_objectness = float(v)
            case "loss_rpn_box_reg":
                loss_rpn_box_reg = float(v)

    return loss_box_reg, loss_classifier, loss_objectness, loss_rpn_box_reg


@wrp
def pass_checkpoint(root_path, model, optimizer, epoch, best_train_loss, best_val_loss, best_map, total_map, model_name,
                    init_exiting_pat, exiting_pat, num_epochs, momentum, device, batch_size, metrics=None,
                    weight_decay=None, lr_scheduler_factor=None, op_id=None, coco_json_path=None,
                    lr_scheduler_patience=None, score_thresh=None, learning_rate=None, anchor_size=None,
                    anchor_ratioe=None, mAP_50=None, mAP_75=None, avg_train_loss=None, avg_val_loss=None, opt=None,
                    avg_train_loss_box_reg=None, avg_train_loss_classifier=None, avg_train_loss_objectness=None,
                    avg_train_loss_rpn_box_reg=None, avg_val_loss_box_reg=None, avg_val_loss_classifier=None,
                    avg_val_loss_objectness=None, avg_val_loss_rpn_box_reg=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_train_loss': best_train_loss,
        'best_val_loss': best_val_loss,
        'best_map': best_map,  # Save the best mAP
        'optimizer': opt,
    }

    hparams_dict, hparams_file = get_hparam_dict(op_id, root_path, coco_json_path, device, model_name, batch_size,
                                                 learning_rate, momentum,
                                                 weight_decay, lr_scheduler_factor, lr_scheduler_patience, score_thresh,
                                                 num_epochs,
                                                 best_map, epoch,
                                                 anchor_size, anchor_ratioe, mAP_50, mAP_75,
                                                 avg_train_loss,
                                                 avg_val_loss, opt, avg_train_loss_box_reg, avg_train_loss_classifier,
                                                 avg_train_loss_objectness, avg_train_loss_rpn_box_reg,
                                                 avg_val_loss_box_reg, avg_val_loss_classifier, avg_val_loss_objectness,
                                                 avg_val_loss_rpn_box_reg)
    save_hparam(hparams_file, hparams_dict)

    if metrics is not None:
        metrics_file = f'{root_path}/progress/metrics/metrics_{model_name}_{opt}_{epoch}.json'
        save_hparam(metrics_file, metrics)

    best_hparams_file = f'{root_path}/progress/best_hparams.json'

    if total_map > best_map:
        best_map = total_map
        checkpoint['best_map'] = best_map
        msg("Best mAP! Saving...")
        torch.save(
            checkpoint, f'{root_path}/progress/fasterrcnn_{model_name}_fpn_{opt}_best_model.pth')
        save_hparam(best_hparams_file, hparams_dict)

    if avg_val_loss > best_val_loss:
        best_val_loss = avg_val_loss
        checkpoint['best_val_loss'] = best_val_loss
        msg("Saving checkpoint...")
        torch.save(checkpoint,
                   f'{root_path}/progress/fasterrcnn_{model_name}_{opt}_fpn_loss.pth')
        exiting_pat = init_exiting_pat
    elif avg_train_loss > best_train_loss:
        best_train_loss = avg_train_loss
        checkpoint['best_train_loss'] = best_train_loss
        msg("Saving checkpoint...")
        torch.save(checkpoint,
                   f'{root_path}/progress/fasterrcnn_{model_name}_{opt}_fpn_loss.pth')
        exiting_pat = init_exiting_pat
    else:
        exiting_pat -= 1
        if not exiting_pat:
            return exiting_pat, best_map

    return exiting_pat, best_map


def format_predictions(annotations, preds, preds_formatted=None, gt_formatted=None):
    id_counter = 0  # Initialize a counter for unique annotation ids
    for i, pred in enumerate(preds):
        for j in range(len(pred['boxes'])):
            if len(pred['masks']) > 0:
                rle = maskUtils.encode(np.asfortranarray(pred['masks'][j].cpu().numpy().astype(np.uint8)))
                segmentation = rle
            else:
                x, y, x2, y2 = map(int, pred['boxes'][j])
                segmentation = [[x, y, x2, y, x2, y2, x, y2]]

            preds_formatted.append({'image_id': i,
                                    'category_id': pred['labels'][j].item(),
                                    'bbox': pred['boxes'][j].tolist(),
                                    'segmentation': segmentation,
                                    'score': pred['scores'][j].item()})

        for j in range(len(annotations[i]['boxes'])):
            if len(annotations[i]['masks']) > 0:
                rle = maskUtils.encode(np.asfortranarray(annotations[i]['masks'][j].cpu().numpy().astype(np.uint8)))
                segmentation = rle
            else:
                x, y, x2, y2 = map(int, annotations[i]['boxes'][j])
                segmentation = [[x, y, x2, y, x2, y2, x, y2]]

            gt_formatted.append({'image_id': i,
                                 'category_id': annotations[i]['labels'][j].item(),
                                 'bbox': annotations[i]['boxes'][j].tolist(),
                                 'area': annotations[i]['area'][j].item(),
                                 'iscrowd': annotations[i]['iscrowd'][j].item(),
                                 'segmentation': segmentation,
                                 'id': id_counter})
            id_counter += 1  # Increment the counter for each annotation

    return preds_formatted, gt_formatted


@wrp
def evaluation(op_id, dataset, single_cat=False, preds_formatted=None, gt_formatted=None):
    # Convert predictions and ground-truths to COCO format
    coco_gt = COCO()

    categories = dataset.categories

    coco_gt.dataset = {'annotations': gt_formatted,
                       'categories': categories}

    # coco_gt.dataset['images'] = [{'id': i} for i in range(len(dataset))]
    coco_gt.dataset['images'] = [{'id': i, 'height': dataset[i][0].shape[1], 'width': dataset[i][0].shape[2]} for i in
                                 range(len(dataset))]
    coco_gt.createIndex()

    if not preds_formatted:
        msg("No predictions made for this batch, adding placeholder prediction.")
        preds_formatted = [
            {'image_id': 0, 'category_id': 0, 'bbox': [0, 0, 1, 1], 'segmentation': [[0, 0, 1, 0, 1, 1, 0, 1]],
             'score': 0.000001}]

    coco_preds = coco_gt.loadRes(preds_formatted)

    # Initialize COCO evaluator
    coco_eval = COCOeval(coco_gt, coco_preds, 'bbox')  # FIXME: should be segm

    # Compute the mAP
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "op_id": op_id,
        # "precision": coco_eval.eval['precision'].tolist(),
        # "recall": coco_eval.eval['recall'].tolist(),
        "AP": coco_eval.stats[0:5].tolist(),
        "AP_small": coco_eval.stats[0],
        "AP_medium": coco_eval.stats[1],
        "AP_large": coco_eval.stats[2],
        "AR_max_1": coco_eval.stats[5],
        "AR_max_10": coco_eval.stats[6],
        "AR_max_100": coco_eval.stats[7],
        "AR_small": coco_eval.stats[8],
        "AR_medium": coco_eval.stats[9],
        "AR_large": coco_eval.stats[10],
    }
    mAP_50 = metrics["AP"][0]
    mAP_75 = metrics["AP"][1]
    # This should now give the mAP over the entire validation set
    total_map = coco_eval.stats[0]

    msg(f"Mean Average Precision: {total_map}")

    return metrics, total_map, mAP_50, mAP_75
