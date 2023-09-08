import cv2
import numpy as np
from matplotlib import cm
from torchvision.transforms import functional as F
from mrcnn.utilities import get_loss_dict, get_losses, format_predictions, \
    get_preds

import torch
from torch.cuda.amp import GradScaler, autocast
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

from tqdm import tqdm

from mrcnn.utilities import wrp, msg
import torch
import cv2
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class MaskRCNNModel:
    def __init__(self, num_classes=4, score_thresh=0.05, backbone_name='resnet50',
                 anchor_size=((16,), (32,), (64,), (128,), (256,)), anchor_ratio=(0.5, 1.0, 2.0), device=None):
        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.backbone_name = backbone_name
        self.anchor_size = anchor_size
        self.anchor_ratio = anchor_ratio
        self.model = self.initialize_model()
        self.model.to(device)
        self.params = self.model.parameters()

    @wrp
    def initialize_model(self):
        msg(f"Initializing pre-trained Mask R-CNN with backbone: FPN+{self.backbone_name}...")

        # Create the backbone
        backbone = resnet_fpn_backbone(self.backbone_name, pretrained=True)

        # Initialize the Mask R-CNN model
        self.model = MaskRCNN(backbone, self.num_classes)

        # Create the anchor generator
        anchor_generator = AnchorGenerator(
            sizes=self.anchor_size, aspect_ratios=(self.anchor_ratio,) * len(self.anchor_size))
        self.model.rpn.anchor_generator = anchor_generator

        # Replace the box predictor
        in_features_box = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, self.num_classes)

        # Replace the mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, self.num_classes)

        # Set the prediction threshold
        self.model.roi_heads.score_thresh = self.score_thresh

        return self.model

    @wrp
    def training(self, epoch, num_epochs, optimizer, data_loader_train, device, evaluate):
        self.model.train()

        preds_formatted = []  # Store all predictions here
        gt_formatted = []  # Store all ground truths here

        scaler = GradScaler()

        tot_loss_box_reg = 0
        tot_loss_classifier = 0
        tot_loss_objectness = 0
        tot_loss_rpn_box_reg = 0
        total_loss = 0

        for imgs, annotations in tqdm(data_loader_train, desc=f"Training epoch {epoch + 1} / {num_epochs} ..."):
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            optimizer.zero_grad()

            with autocast():
                loss_dict = self.model(imgs, annotations)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_box_reg, loss_classifier, loss_objectness, loss_rpn_box_reg = get_losses(loss_dict)

            tot_loss_box_reg += loss_box_reg
            tot_loss_classifier += loss_classifier
            tot_loss_objectness += loss_objectness
            tot_loss_rpn_box_reg += loss_rpn_box_reg
            total_loss += losses.item()

            if evaluate:
                preds = get_preds(self.model, imgs)
                preds_formatted, gt_formatted = format_predictions(annotations, preds, preds_formatted, gt_formatted)

        avg_train_loss = total_loss / len(data_loader_train)
        msg(f"Training Loss: {avg_train_loss}")
        avg_train_loss_box_reg = tot_loss_box_reg / len(data_loader_train)
        avg_train_loss_classifier = tot_loss_classifier / len(data_loader_train)
        avg_train_loss_objectness = tot_loss_objectness / len(data_loader_train)
        avg_train_loss_rpn_box_reg = tot_loss_rpn_box_reg / len(data_loader_train)

        return self.model, avg_train_loss, preds_formatted, gt_formatted, avg_train_loss_box_reg, avg_train_loss_classifier, avg_train_loss_objectness, avg_train_loss_rpn_box_reg

    @wrp
    def validation(self, data_loader_val, epoch, device):
        self.model.eval()
        total_val_loss = 0

        preds_formatted = []  # Store all predictions here
        gt_formatted = []  # Store all ground truths here
        tot_loss_box_reg = 0
        tot_loss_classifier = 0
        tot_loss_objectness = 0
        tot_loss_rpn_box_reg = 0

        with torch.no_grad():
            for imgs, annotations in tqdm(data_loader_val, desc=f"Validation epoch {epoch + 1}..."):
                imgs = list(img.to(device) for img in imgs)
                annotations = [
                    {k: v.to(device) for k, v in t.items()} for t in annotations]

                with autocast():
                    preds = self.model(imgs)
                    loss_dict = get_loss_dict(self.model, imgs, annotations)
                    losses = sum(loss for loss in loss_dict.values())

                total_val_loss += losses.item()
                loss_box_reg, loss_classifier, loss_objectness, loss_rpn_box_reg = get_losses(loss_dict)
                tot_loss_box_reg += loss_box_reg
                tot_loss_classifier += loss_classifier
                tot_loss_objectness += loss_objectness
                tot_loss_rpn_box_reg += loss_rpn_box_reg

                preds_formatted, gt_formatted = format_predictions(annotations, preds, preds_formatted, gt_formatted)

        avg_val_loss = total_val_loss / len(data_loader_val)
        msg(f"Validation Loss: {avg_val_loss}")

        avg_val_loss_box_reg = tot_loss_box_reg / len(data_loader_val)
        avg_val_loss_classifier = tot_loss_classifier / len(data_loader_val)
        avg_val_loss_objectness = tot_loss_objectness / len(data_loader_val)
        avg_val_loss_rpn_box_reg = tot_loss_rpn_box_reg / len(data_loader_val)

        return self.model, avg_val_loss, preds_formatted, gt_formatted, avg_val_loss_box_reg, avg_val_loss_classifier, avg_val_loss_objectness, avg_val_loss_rpn_box_reg

    def inference(self, image_path, device="cuda:0", min_confidence=0.33, categories=None):

        # Create a mapping from IDs to names
        id_to_name = {cat['id']: cat['name'] for cat in categories}

        # Define color mapping
        color_map = ['black', 'red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'purple', 'orange', 'pink']

        # Load image from path and convert to PIL Image
        image = Image.open(image_path).convert("RGB")

        # Convert to tensor and normalize
        img_tensor = F.to_tensor(image).unsqueeze(0).to(device)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(img_tensor)

        # Convert prediction to CPU
        prediction = {k: v.cpu() for k, v in prediction[0].items()}

        # Filter by score threshold
        high_conf_indices = torch.where(prediction['scores'] > min_confidence)[0]
        if len(high_conf_indices) == 0:
            print("No high confidence predictions")
            return

        # Filter predictions
        filtered_prediction = {k: v[high_conf_indices] for k, v in prediction.items()}

        # Display image and annotations
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for i in range(len(filtered_prediction['boxes'])):
            box = filtered_prediction['boxes'][i].numpy().astype(int)
            label_id = filtered_prediction['labels'][i].item()
            label_name = id_to_name.get(label_id, str(label_id))
            score = filtered_prediction['scores'][i].item()
            mask = filtered_prediction['masks'][i, 0].numpy()

            # Draw bounding box
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1,
                                     edgecolor=color_map[label_id % len(color_map)],
                                     facecolor='none')
            ax.add_patch(rect)

            # Label
            plt.text(box[0], box[1], f"{label_name}: {score:.2f}", color=color_map[label_id % len(color_map)])

            # Mask to Polygon
            contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour_squeezed = contour.squeeze()
                if len(contour_squeezed.shape) == 2 and contour_squeezed.shape[1] == 2:
                    poly_patch = patches.Polygon(contour_squeezed, closed=True,
                                                 edgecolor=color_map[label_id % len(color_map)], facecolor='none')
                    ax.add_patch(poly_patch)
                else:
                    print(f"Skipping contour with unexpected shape: {contour_squeezed.shape}")

        plt.show()


