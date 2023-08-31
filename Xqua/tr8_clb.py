import datetime
import json
import os
import uuid
from pathlib import Path
from pprint import pprint

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.transforms import transforms
from tqdm import tqdm

DATASETS = [
    r"C:\Users\LTSS2023\Documents\elhadjmb\datasets\div_1",
    r"C:\Users\LTSS2023\Documents\elhadjmb\datasets\blc_55",
    r"C:\Users\LTSS2023\Documents\elhadjmb\datasets\blc_11",
    r"C:\Users\LTSS2023\Documents\elhadjmb\datasets\chegg",
    r"C:\Users\LTSS2023\Documents\elhadjmb\datasets\train",
r"C:\Users\LTSS2023\Documents\elhadjmb\datasets\blc_13",
r"C:\Users\LTSS2023\Documents\elhadjmb\datasets\blc_0",

]

MODELS = [
    "resnet50",
    "resnet101",
    "resnet152",
]


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
def calc_mean_std(coco_path):
    msg("Calculating Mean and Std ...")
    imageFilesDir = Path(str(os.path.dirname(coco_path)))

    files = list(imageFilesDir.rglob('*.jpg'))
    len(files)

    mean = np.array([0., 0., 0.])
    stdTemp = np.array([0., 0., 0.])

    numSamples = len(files)

    for i in tqdm(range(numSamples)):
        im = cv2.imread(str(files[i]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(float) / 255.

        for j in range(3):
            mean[j] += np.mean(im[:, :, j])

    mean = list(mean / numSamples)
    msg(f"MEAN: {mean}")
    for i in tqdm(range(numSamples)):
        im = cv2.imread(str(files[i]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(float) / 255.
        for j in range(3):
            stdTemp[j] += ((im[:, :, j] - mean[j]) ** 2).sum() / \
                          (im.shape[0] * im.shape[1])

    std = list(np.sqrt(stdTemp / numSamples))
    msg(f"STD: {std}")
    return mean, std
@wrp
def get_optimal_anchors(annotation_file, num_sizes=5, num_ratios=3):
    with open(annotation_file, 'r') as file:
        data = json.load(file)

    # Extract bounding box sizes and aspect ratios
    bbox_sizes = []
    aspect_ratios = []
    for annotation in data['annotations']:
        bbox = annotation['bbox']
        width = bbox[2]
        height = bbox[3]
        bbox_sizes.append(np.sqrt(width * height))  # Use square root of area to approximate side length
        aspect_ratios.append(width / height)

    bbox_sizes = np.array(bbox_sizes).reshape(-1, 1)
    aspect_ratios = np.array(aspect_ratios).reshape(-1, 1)

    # Perform k-means clustering to find optimal anchor sizes and ratios
    kmeans_sizes = KMeans(n_clusters=num_sizes, random_state=0).fit(bbox_sizes)
    kmeans_ratios = KMeans(n_clusters=num_ratios, random_state=0).fit(aspect_ratios)

    # Extract the cluster centers as the optimal anchor sizes and ratios
    optimal_sizes = tuple(tuple(map(int, center)) for center in sorted(list(kmeans_sizes.cluster_centers_)))
    optimal_ratios = tuple(sorted(kmeans_ratios.cluster_centers_.flatten().tolist()))

    return optimal_sizes, optimal_ratios


class CocoDataset(Dataset):
    def __init__(self, coco_json_path: str, transform=None):
        try:
            self.coco = COCO(coco_json_path)
            self.transform = transform
            self.img_ids = self.coco.getImgIds()
            self.images_directory = str(os.path.dirname(coco_json_path))
            msg(f"CocoDataset initialized with {len(self.img_ids)} images.")
        except Exception as e:
            raise Exception(f"Error initializing COCO with error: {e}")

    def __getitem__(self, idx):
        img = target = None
        try:
            img_info = self.coco.loadImgs(self.img_ids[idx])[0]
            img_dir = os.path.join(self.images_directory,
                                   img_info['file_name'])
            img = Image.open(img_dir).convert('RGB')
        except (FileNotFoundError, IndexError, OSError) as e:
            raise Exception(f"Error loading image {img_dir} with error {e}")

        try:
            img_width, img_height = img.size
            img = self.transform(img)
            ann_ids = self.coco.getAnnIds(imgIds=self.img_ids[idx])
            anns = self.coco.loadAnns(ann_ids)
            boxes = [ann['bbox'] for ann in anns if ann['iscrowd'] == 0]
            # Convert COCO format to [x_min, y_min, x_max, y_max]
            boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]]
                     for box in boxes]
            # Ensure boxes don't exceed image dimensions
            boxes = [[max(0, box[0]), max(0, box[1]), min(
                img_width, box[2]), min(img_height, box[3])] for box in boxes]
            # Filter out boxes with non-positive height or width
            boxes = [box for box in boxes if box[2]
                     > box[0] and box[3] > box[1]]
            labels = [ann['category_id']
                      for ann in anns if ann['iscrowd'] == 0]
            areas = [ann['area'] for ann in anns if ann['iscrowd'] == 0]
            # Match the boxes to their corresponding labels and areas
            labels = [label for box, label in zip(
                boxes, labels) if box[2] > box[0] and box[3] > box[1]]
            areas = [area for box, area in zip(
                boxes, areas) if box[2] > box[0] and box[3] > box[1]]
        except Exception as e:
            raise Exception(
                f"Error processing annotations for image {img_dir} with error {e}")

        try:
            if len(boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.empty((0,), dtype=torch.int64)
                areas = torch.empty((0,), dtype=torch.float32)
            else:
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
                areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        except TypeError as e:
            raise Exception(f"Error during tensor conversion with error: {e}")

        try:
            target = {"boxes": boxes,
                      "labels": labels,
                      "image_id": torch.tensor([self.img_ids[idx]]),
                      "area": areas,
                      "iscrowd": iscrowd}
            return img, target
        except Exception as e:
            raise Exception(
                f"Error returning image and target with error: {e}")
        finally:
            return img, target

    def __len__(self):
        try:
            return len(self.img_ids)
        except Exception as e:
            raise Exception(f"Error getting dataset length with error: {e}")

# ║ Applying normalization with mean: [] and std: [] ║
@wrp
def get_transform(coco_path=None, mean=(0.6349435724011382, 0.5683341293092785, 0.5390255380004837),
                  std=(0.13469885 ,0.13852801, 0.15647008)):
    if coco_path is not None:
        msg("Calculating Mean and Std ...")
        imageFilesDir = Path(str(os.path.dirname(coco_path)))

        files = list(imageFilesDir.rglob('*.jpg'))
        len(files)

        mean = np.array([0., 0., 0.])
        stdTemp = np.array([0., 0., 0.])
        std = np.array([0., 0., 0.])

        numSamples = len(files)

        for i in range(numSamples):
            im = cv2.imread(str(files[i]))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(float) / 255.

            for j in range(3):
                mean[j] += np.mean(im[:, :, j])

        mean = list(mean / numSamples)
        for i in range(numSamples):
            im = cv2.imread(str(files[i]))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(float) / 255.
            for j in range(3):
                stdTemp[j] += ((im[:, :, j] - mean[j]) ** 2).sum() / \
                              (im.shape[0] * im.shape[1])

        std = np.sqrt(stdTemp / numSamples)

    try:
        msg(f"Applying normalization with mean: {mean} and std: {std}")
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    except Exception as e:
        raise Exception(f"Error creating transformation with error: {e}")


def get_model(num_classes=4, score_thresh=0.05, backbone_name='resnet50',
              anchor_size=((16,), (32,), (64,), (128,), (256,)), anchor_ratio=(0.5, 1.0, 2.0)):
    try:
        msg(
            f"Initializing pre-trained Faster R-CNN with backbone: FPN+{backbone_name}...")

        backbone = resnet_fpn_backbone(backbone_name, pretrained=True)
        model = FasterRCNN(backbone, num_classes, box_detections_per_img=2000)
        anchor_generator = AnchorGenerator(
            sizes=anchor_size, aspect_ratios=(anchor_ratio,) * len(anchor_size))
        model.rpn.anchor_generator = anchor_generator

        # Replace the box predictor with a new one
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)

        # Lower the prediction threshold
        model.roi_heads.score_thresh = score_thresh
    except Exception as e:
        raise Exception(f"Error initializing model with error: {e}")

    return model


def oget_model(num_classes=5, score_thresh=0.05, backbone_name='resnet50',
               anchor_size=((16,), (32,), (64,), (128,), (256,), (512,), (1024,)),
               anchor_ratio=(0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)):
    msg(f"Initializing pre-trained Faster R-CNN with backbone: FPN+{backbone_name}...")

    backbone = resnet_fpn_backbone(backbone_name, pretrained=True)
    model = FasterRCNN(backbone, num_classes, box_detections_per_img=2000)

    # Create the new anchor generator
    anchor_generator = AnchorGenerator(sizes=anchor_size, aspect_ratios=anchor_ratio)

    # Create the RPN head
    in_channels = backbone.out_channels  # depends on the backbone
    rpn_head = RPNHead(in_channels, anchor_generator.num_anchors_per_location()[0])

    # Replace the RPN and the box predictor
    model.rpn.anchor_generator = anchor_generator
    model.rpn.head = rpn_head

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Lower the prediction threshold
    model.roi_heads.score_thresh = score_thresh

    return model


def xget_model(num_classes=5, score_thresh=0.05, backbone_name='resnet50',
               anchor_size=((16,), (32,), (64,), (128,), (256,), (512,), (1024,)),
               anchor_ratio=(0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)):
    msg(f"Initializing pre-trained Faster R-CNN with backbone: FPN+{backbone_name}...")

    backbone = resnet_fpn_backbone(backbone_name, pretrained=True)
    model = FasterRCNN(backbone, num_classes, box_detections_per_img=2000)

    # Create the new anchor generator
    anchor_generator = AnchorGenerator(sizes=anchor_size, aspect_ratios=anchor_ratio)

    # Create the RPN head
    in_channels = backbone.out_channels  # depends on the backbone
    rpn_head = RPNHead(in_channels, anchor_generator.num_anchors_per_location()[0])

    # Replace the RPN and the box predictor
    model.rpn.anchor_generator = anchor_generator
    model.rpn.head = rpn_head

    # Specify the feature maps to use
    feature_map_names = ['0', '1', '2', '3', '4']  # adjust this to select the feature maps you want to use

    # Modify the RPN and ROI heads to use the specified feature maps
    model.rpn = torchvision.models.detection.rpn.RegionProposalNetwork(
        anchor_generator, rpn_head, 0.7, 2000, 2000,
        tuple(int(fm) for fm in feature_map_names)
    )
    model.roi_heads = torchvision.models.detection.roi_heads.RoIHeads(
        # Box
        torchvision.models.detection.roi_heads.MultiScaleRoIAlign(
            feature_map_names, 7, 2),
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(256, num_classes),
        # Mask
        torchvision.models.detection.roi_heads.MultiScaleRoIAlign(
            feature_map_names, 14, 2),
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(256, num_classes),
        0.5, 0.5, 2000, 0.25
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Lower the prediction threshold
    model.roi_heads.score_thresh = score_thresh

    return model


def collate_fn(batch):
    try:
        result = tuple(zip(*batch))
        return result
    except Exception as e:
        raise Exception(f"Error collating batch: {e}")


def save_hparam(hparams_file, hparam_dict):
    msg(f"Saving: {hparams_file}")
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
def evaluation(op_id, dataset, single_cat=False, preds_formatted=None, gt_formatted=None):
    # Convert predictions and ground-truths to COCO format
    coco_gt = COCO()

    categories = [{"id": 1, "name": "Eggs", "supercategory": "none"},
                  {"id": 2, "name": "egg", "supercategory": "Eggs"}] if single_cat else [
        {"id": 1, "name": "Eggs", "supercategory": "none"},
        {"id": 2, "name": "blastula_gastrula", "supercategory": "Eggs"}, {
            "id": 3, "name": "cleavage", "supercategory": "Eggs"},
        {"id": 4, "name": "organogenesis", "supercategory": "Eggs"}]

    coco_gt.dataset = {'annotations': gt_formatted,
                       'categories': categories}

    coco_gt.dataset['images'] = [{'id': i} for i in range(len(dataset))]
    coco_gt.createIndex()

    if not preds_formatted:
        msg("No predictions made for this batch, adding placeholder prediction.")
        preds_formatted = [
            {'image_id': 0, 'category_id': 0, 'bbox': [0, 0, 1, 1], 'score': 0.000001}]

    coco_preds = coco_gt.loadRes(preds_formatted)

    # Initialize COCO evaluator
    coco_eval = COCOeval(coco_gt, coco_preds, 'bbox')

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
    # Format the predictions and ground-truths
    id_counter = 0  # Initialize a counter for unique annotation ids
    for i, pred in enumerate(preds):
        for j in range(len(pred['boxes'])):
            preds_formatted.append({'image_id': i, 'category_id': pred['labels'][j].item(),
                                    'bbox': pred['boxes'][j].tolist(),
                                    'score': pred['scores'][j].item()})

        for j in range(len(annotations[i]['boxes'])):
            gt_formatted.append({'image_id': i, 'category_id': annotations[i]['labels'][j].item(),
                                 'bbox': annotations[i]['boxes'][j].tolist(),
                                 'area': annotations[i]['area'][j].item(),
                                 'iscrowd': annotations[i]['iscrowd'][j].item(),
                                 'id': id_counter})  # Add the unique id to each annotation
            id_counter += 1  # Increment the counter for each annotation

    return preds_formatted, gt_formatted


def get_loss_dict(model, imgs, annotations):
    model.train()
    ld = model(imgs, annotations)
    model.eval()
    return ld


@wrp
def validation(model, data_loader_val, epoch, device):
    model.eval()
    total_val_loss = 0

    preds_formatted = []  # Store all predictions here
    gt_formatted = []  # Store all ground truths here
    tot_loss_box_reg = 0
    tot_loss_classifier = 0
    tot_loss_objectness = 0
    tot_loss_rpn_box_reg = 0
    with torch.no_grad():  # Turn off gradients for validation
        for imgs, annotations in tqdm(data_loader_val):
            try:
                imgs = list(img.to(device) for img in imgs)
                annotations = [
                    {k: v.to(device) for k, v in t.items()} for t in annotations]

                # Generate predictions
                preds = model(imgs)

                # Calculate validation loss
                loss_dict = get_loss_dict(model, imgs, annotations)
                loss_box_reg, loss_classifier, loss_objectness, loss_rpn_box_reg = get_losses(loss_dict)
                losses = sum(loss for loss in loss_dict.values())
                total_val_loss += losses.item()
                tot_loss_box_reg += loss_box_reg
                tot_loss_classifier += loss_classifier
                tot_loss_objectness += loss_objectness
                tot_loss_rpn_box_reg += loss_rpn_box_reg
                preds_formatted, gt_formatted = format_predictions(annotations, preds, preds_formatted, gt_formatted)

            except Exception as e:
                raise Exception(
                    f"Error occurred during validation in epoch {epoch + 1}. Error: {e}")
    avg_val_loss = total_val_loss / len(data_loader_val)
    msg(f"Validation Loss: {avg_val_loss}")
    avg_val_loss_box_reg = tot_loss_box_reg / len(data_loader_val)
    avg_val_loss_classifier = tot_loss_classifier / len(data_loader_val)
    avg_val_loss_objectness = tot_loss_objectness / len(data_loader_val)
    avg_val_loss_rpn_box_reg = tot_loss_rpn_box_reg / len(data_loader_val)
    return model, avg_val_loss, preds_formatted, gt_formatted, avg_val_loss_box_reg, avg_val_loss_classifier, avg_val_loss_objectness, avg_val_loss_rpn_box_reg


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
def training(model, epoch, num_epochs, optimizer, data_loader_train, device, evaluate):
    model.train()

    preds_formatted = []  # Store all predictions here
    gt_formatted = []  # Store all ground truths here

    tot_loss_box_reg = 0
    tot_loss_classifier = 0
    tot_loss_objectness = 0
    tot_loss_rpn_box_reg = 0
    total_loss = 0
    msg(f"Epoch {epoch + 1} / {num_epochs}")
    for imgs, annotations in tqdm(data_loader_train):
        try:
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()}
                           for t in annotations]

            optimizer.zero_grad()

            loss_dict = model(imgs, annotations)
            loss_box_reg, loss_classifier, loss_objectness, loss_rpn_box_reg = get_losses(loss_dict)
            losses = sum(loss for loss in loss_dict.values())

            tot_loss_box_reg += loss_box_reg
            tot_loss_classifier += loss_classifier
            tot_loss_objectness += loss_objectness
            tot_loss_rpn_box_reg += loss_rpn_box_reg
            total_loss += losses.item()

            losses.backward()
            optimizer.step()

            if evaluate:
                preds = get_preds(model, imgs)
                preds_formatted, gt_formatted = format_predictions(annotations, preds, preds_formatted, gt_formatted)


        except Exception as e:
            raise Exception(
                f"Error occurred during training in epoch {epoch + 1}. Error: {e}")

    avg_train_loss = total_loss / len(data_loader_train)
    msg(f"Training Loss: {avg_train_loss}")
    avg_train_loss_box_reg = tot_loss_box_reg / len(data_loader_train)
    avg_train_loss_classifier = tot_loss_classifier / len(data_loader_train)
    avg_train_loss_objectness = tot_loss_objectness / len(data_loader_train)
    avg_train_loss_rpn_box_reg = tot_loss_rpn_box_reg / len(data_loader_train)

    return model, avg_train_loss, preds_formatted, gt_formatted, avg_train_loss_box_reg, avg_train_loss_classifier, avg_train_loss_objectness, avg_train_loss_rpn_box_reg


@wrp
def display_hparams(op_id=None, root_path=None, coco_json_path=None, device=None, model_name=None, batch_size=None,
                    learning_rate=None, momentum=None, weight_decay=None, lr_scheduler_factor=None,
                    lr_scheduler_patience=None, score_thresh=None):
    hparam_dict, _ = get_hparam_dict(op_id, root_path, coco_json_path, device, model_name, batch_size, learning_rate,
                                     momentum,
                                     weight_decay, lr_scheduler_factor, lr_scheduler_patience, score_thresh, full=False)
    pprint(hparam_dict)


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


@wrp
def start(coco_json_path, num_classes=5, train_val_split=0.8, resume_from_checkpoint=None, batch_size=1,
          learning_rate=0.005, momentum=0.9, weight_decay=0.0005, lr_scheduler_factor=0.1, lr_scheduler_patience=3,
          num_epochs=20, score_thresh=0.05, model_name="resnet101", anchor_size=None, anchor_ratio=None,
          single_cat=False, opt="SGD", mean=None,std=None):
    root_path = str(os.path.dirname(
        str(os.path.dirname(str(os.path.dirname(coco_json_path))))))
    num_classes = 3 if single_cat else 5
    # Getting transforms...
    transform = get_transform(coco_path=None, mean=mean, std=std)

    # Creating dataset...
    try:
        dataset = CocoDataset(coco_json_path, transform=transform)
    except Exception as e:
        raise Exception(f"Failed to create dataset. Error: {e}")

    # Splitting dataset into training and validation sets...
    try:
        train_size = int(train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size])
    except Exception as e:
        raise Exception(f"Failed to split dataset. Error: {e}")

    # Creating data loaders...
    try:
        data_loader_train = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        data_loader_val = DataLoader(
            val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    except Exception as e:
        raise Exception(f"Failed to create data loaders. Error: {e}")

    # Setting device...
    try:
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
    except Exception as e:
        raise Exception(f"Failed to set device. Error: {e}")

    # Getting model...
    try:

        model = get_model(num_classes, score_thresh,
                          model_name, anchor_size, anchor_ratio)
        model.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        if opt.upper() == 'SGD':
            optimizer = torch.optim.SGD(
                params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)
        elif opt.upper() == 'ADAM':
            optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999),
                                         eps=1e-08)
    except Exception as e:
        raise Exception(f"Failed to get model or set optimizer. Error: {e}")

    # Setting learning rate scheduler...
    try:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_scheduler_factor, patience=lr_scheduler_patience, verbose=True)
    except Exception as e:
        raise Exception(f"Failed to set learning rate scheduler. Error: {e}")

    init_exiting_pat = 5
    exiting_pat = init_exiting_pat

    model, optimizer, start_epoch, best_map, best_train_loss, best_val_loss = load_checkpoint(root_path, model_name,
                                                                                              model, optimizer, opt,
                                                                                              num_classes)

    for epoch in range(start_epoch, num_epochs):
        op_id = uuid.uuid4()

        display_hparams(op_id, root_path, coco_json_path, device, model_name, batch_size,
                        learning_rate, momentum, weight_decay, lr_scheduler_factor,
                        lr_scheduler_patience, score_thresh)

        model, avg_train_loss, preds_formatted, gt_formatted, avg_train_loss_box_reg, avg_train_loss_classifier, avg_train_loss_objectness, avg_train_loss_rpn_box_reg = training(
            model, epoch, num_epochs, optimizer,
            data_loader_train, device, evaluate=False)

        # metrics, total_map, mAP_50, mAP_75 = evaluation(op_id, dataset, single_cat, preds_formatted, gt_formatted)

        model, avg_val_loss, preds_formatted, gt_formatted, avg_val_loss_box_reg, avg_val_loss_classifier, avg_val_loss_objectness, avg_val_loss_rpn_box_reg = validation(
            model, data_loader_val, epoch, device)

        metrics, total_map, mAP_50, mAP_75 = evaluation(op_id, dataset, single_cat, preds_formatted, gt_formatted)

        exiting_pat, best_map = pass_checkpoint(root_path, model, optimizer, epoch, best_train_loss, best_val_loss,
                                                best_map, total_map, model_name,
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


@wrp
def level(number, dataset, model, single_cat):
    json_file = dataset + r"\sc_annotations.json" if single_cat else dataset + r"\annotations.json"
    batch_size = 16  # 2 if model == "resnet152" else 4 if model == "resnet101" else 8
    learning_rate = 1e-3  # 0.012
    momentum = 0.78
    weight_decay = 0.0001
    lr_scheduler_factor = 0.9
    lr_scheduler_patience = 8
    score_thresh = 0.001
    num_epochs = 300 * number
    train_val_split = 0.8
    anchors = get_optimal_anchors(json_file)
    anchor_size = anchors[0]  # ((28,), (54,), (80,), (106,), (132,))
    # ((10,), (26,), (42,), (57,), (73,)) for blc7  # ((16,), (32,), (48,), (64,), (80,))
    anchor_ratio = anchors[1]  # (0.5998266656490321, 0.9571966923446008, 1.3145667190401695)
    # (0.5624379447109308, 0.9246200011074786, 1.2868020575040262)  # (0.78, 1.0, 1.28)
    opt = "SGD"
    mean, std = calc_mean_std(json_file)


    time_started = datetime.datetime.now()
    while batch_size > 0:
        try:
            start(
                json_file,
                batch_size=batch_size,
                train_val_split=train_val_split,
                learning_rate=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                lr_scheduler_factor=lr_scheduler_factor,
                lr_scheduler_patience=lr_scheduler_patience,
                num_epochs=num_epochs,
                score_thresh=score_thresh,
                model_name=model,
                anchor_size=anchor_size,
                anchor_ratio=anchor_ratio,
                single_cat=single_cat,
                opt=opt,
                mean=mean,
                std = std
            )
        except Exception as e:
            if "CUDA out of memory" in str(e):
                msg(f"CUDA error likely, batch_size: {batch_size}, reducing...")
                batch_size //= 2
                continue
            else:
                raise e
        break
    if batch_size < 1:
        raise Exception(f"YOUR {model} MODEL IS TOO BIG")
    time_ended = datetime.datetime.now()
    level_p = {"time_started": time_started, "time_ended": time_ended, "level": number, "model": model,
               "dataset": dataset,
               "single": single_cat}
    location = r"C:\Users\LTSS2023\Documents\elhadjmb/progress/level.json"
    save_hparam(hparams_file=location, hparam_dict=level_p)


if __name__ == '__main__':

    for number in range(1, 2 + 1):
        dataset = DATASETS[1] # 11 9


        single_cat = (True if (number == 1) else False) and True

        model = MODELS[1]
        level(number=number, model=model, dataset=dataset, single_cat=single_cat)
