import os
import random
from pprint import pprint
from timeit import timeit

import cv2
import numpy as np
import torch
import json

from matplotlib import ticker
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F
from torchvision.ops import nms
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def plot_value_counts(values, act=False):
    # Round values to the nearest two decimal places
    if act:
        for i, v in enumerate(values):
            t = v / 100
            values[i] = t
        values = values[:len(values) // 2 + 1]
    values = np.round(values, 2)

    # Count the occurrences of each value
    unique, counts = np.unique(values, return_counts=True)

    # Sort the unique values and their counts in ascending order
    sorted_indices = np.argsort(unique)
    unique = unique[sorted_indices]
    counts = counts[sorted_indices]

    # Determine the figure size based on the number of unique values
    width = 20
    height = 6

    # Plot a bar chart of the values versus their counts
    fig, ax = plt.subplots(figsize=(width, height))
    ax.bar(unique, counts)

    # Increase the number of x-axis ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=min(20, len(unique))))

    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.set_title('Value Counts')
    plt.show()


def analyze_annotations(input_json_path):
    with open(input_json_path) as f:
        data = json.load(f)

    annotations = data['annotations']
    images = data['images']

    # Create a dictionary to count the number of annotations per image
    annotations_per_image = {image['id']: 0 for image in images}
    for anno in annotations:
        annotations_per_image[anno['image_id']] += 1

    # Convert to a list and calculate statistics
    annotations_per_image_counts = list(annotations_per_image.values())
    annotations_per_image_counts = np.array(annotations_per_image_counts)

    bbox_sizes = []
    aspect_ratios = []
    bbox_heights = []
    bbox_widths = []

    for anno in annotations:
        bbox = anno['bbox']
        # bbox = [x,y,width,height]
        width = bbox[2]
        height = bbox[3]
        bbox_sizes.append(width * height)
        aspect_ratios.append(float(width) / height)
        bbox_heights.append(height)
        bbox_widths.append(width)

    bbox_sizes = np.array(bbox_sizes)
    aspect_ratios = np.array(aspect_ratios)
    bbox_heights = np.array(bbox_heights)
    bbox_widths = np.array(bbox_widths)

    # plot_value_counts(bbox_sizes, True)
    # plot_value_counts(aspect_ratios)
    plot_value_counts(bbox_sizes)
    plot_value_counts(aspect_ratios)

    analysis_results = {
        'bbox': {
            'count': len(bbox_sizes),
            'size': {
                'max': bbox_sizes.max(),
                'min': bbox_sizes.min(),
                'mean': bbox_sizes.mean(),
                'std': bbox_sizes.std()
            },
            'aspect_ratio': {
                'max': aspect_ratios.max(),
                'min': aspect_ratios.min(),
                'mean': aspect_ratios.mean(),
                'std': aspect_ratios.std()
            }
        },
        'height': {
            'max': bbox_heights.max(),
            'min': bbox_heights.min(),
            'mean': bbox_heights.mean(),
            'std': bbox_heights.std()
        },
        'width': {
            'max': bbox_widths.max(),
            'min': bbox_widths.min(),
            'mean': bbox_widths.mean(),
            'std': bbox_widths.std()
        },
        'annotations_per_image': {
            'max': annotations_per_image_counts.max(),
            'min': annotations_per_image_counts.min(),
            'mean': annotations_per_image_counts.mean(),
            'std': annotations_per_image_counts.std()
        }
    }

    return analysis_results


def display(image):
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Display the image
    plt.imshow(image)
    plt.show()


def draw_predictions(image: np.ndarray, prediction: Dict[str, torch.Tensor]) -> np.ndarray:
    colors = [(0, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 225, 255)]
    for box, label in zip(prediction['boxes'], prediction['labels']):
        color = colors[label.item() % len(colors)]
        cv2.rectangle(image, (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item())),
                      color, 2)
        cv2.putText(image, str(label.item()), (int(box[0].item()), int(box[1].item()) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image


def to_json(stats: Dict[str, float]) -> str:
    return json.dumps(stats, default=str, indent=2)


class InferencePipeline:
    def __init__(self, model: torch.nn.Module, device: str = 'cpu', alpha: float = 1.0, beta: int = 0,
                 mean=None,
                 std=None, nms_threshold: float = 0.75,
                 kernel: np.ndarray = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])):
        if std is None:
            std = [0.229, 0.224, 0.225]
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        self.model = model
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.mean = mean
        self.std = std
        self.nms_threshold = 0
        self.confidence_threshold = 0
        self.kernel = kernel
        self.noise_threshold: float = 0.1
        self.filter_type: str = ('gaussian', 'median')[0]
        self.kernel_size: int = 5

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        # Contrast and brightness control
        enhanced_image = cv2.convertScaleAbs(image, alpha=self.alpha, beta=self.beta)
        # display(enhanced_image)

        # Calculate noise metric (e.g., standard deviation of the image)
        noise_metric = np.std(enhanced_image)

        # Conditionally apply sharpness enhancement
        if noise_metric < self.noise_threshold:
            enhanced_image = cv2.filter2D(enhanced_image, -1, self.kernel)
            # display(enhanced_image)

        # Apply noise reduction filter
        if self.filter_type == 'gaussian':
            enhanced_image = cv2.GaussianBlur(enhanced_image, (self.kernel_size, self.kernel_size), 0)
            # display(enhanced_image)
        elif self.filter_type == 'median':
            enhanced_image = cv2.medianBlur(enhanced_image, self.kernel_size)
            # display(enhanced_image)

        return enhanced_image

    def normalize_image(self, image: np.ndarray) -> torch.Tensor:
        # Convert the image from OpenCV BGR format to PyTorch RGB format and normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # display(image)

        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image

    def infer(self, image: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model([image.to(self.device)])
        # Filter out detections with a confidence score below the threshold
        for p in prediction:
            mask = p['scores'] >= self.confidence_threshold
            p['boxes'] = p['boxes'][mask]
            p['labels'] = p['labels'][mask]
            p['scores'] = p['scores'][mask]
            # Skip if no boxes remain after filtering
            if p['boxes'].numel() == 0:
                continue
        return prediction

    def apply_nms(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        keep = nms(boxes, scores, self.nms_threshold)
        return keep

    def calculate_statistics(self, prediction: Dict[str, torch.Tensor]) -> dict[
        str, dict[str, Any] | list[dict[str, int | Any]]]:
        stats = {}
        summary = {}
        classes = []
        if prediction['labels'].numel() == 0:
            summary['total_count'] = 0
            summary['overall_avg_confidence'] = 0
            stats['summary'] = summary
            return stats
        num_classes = prediction['labels'].max().item() + 1
        for c in range(num_classes):
            mask = prediction['labels'] == c
            class_x = {}
            class_x["id"] = c
            class_x[f'count'] = mask.sum().item()
            class_x[f'avg_confidence'] = prediction['scores'][mask].mean().item()
            classes.append(class_x)

        summary['total_count'] = prediction['labels'].numel()
        summary['overall_avg_confidence'] = prediction['scores'].mean().item()

        stats['summary'] = summary
        stats["classes"] = classes

        return stats

    def run(self, image: np.ndarray) -> str:
        original_image = image
        image = self.enhance_image(image)
        image = self.normalize_image(image)
        prediction = self.infer(image)
        keep = self.apply_nms(prediction[0]['boxes'], prediction[0]['scores'])
        final_prediction = {k: v[keep] for k, v in prediction[0].items()}
        stats = self.calculate_statistics(final_prediction)
        json_stats = to_json(stats)
        drawn_image = draw_predictions(original_image, final_prediction)
        return json_stats, drawn_image


def get_model(path=r"C:\Users\LTSS2023\Documents\elhadjmb/progress/fasterrcnn_resnet50_fpn_SGD_best_model.pth",
              backbone_name="resnet50", num_classes=5, anchor_size=((28,), (54,), (80,), (106,), (132,),),
              anchor_ratio=(0.599, 0.957, 1.314), score_thresh=0.01):
    try:
        print(
            f"Initializing pre-trained Faster R-CNN with backbone: FPN+{backbone_name}...")

        backbone = resnet_fpn_backbone(backbone_name, pretrained=True)
        model = FasterRCNN(backbone, num_classes, box_detections_per_img=2000)
        anchor_generator = AnchorGenerator(
            sizes=anchor_size, aspect_ratios=(anchor_ratio,) * len(anchor_size))
        model.rpn.anchor_generator = anchor_generator

        # Lower the prediction threshold
        model.roi_heads.score_thresh = score_thresh

        print("Loading checkpoint...")
        try:
            checkpoint = torch.load(path)
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

        except Exception as e:
            raise Exception(f"Failed to load checkpoint. Error: {e}")

    except Exception as e:
        raise Exception(f"Error initializing model with error: {e}")

    return model


pipeline = InferencePipeline(get_model(), nms_threshold=0)

# Load your image
# image1 = cv2.imread("/home/hadjm/Videos/20230510_180849.jpg")
# image2 = cv2.imread(
#     "/home/hadjm/Downloads/strexaqua.v11i.coco/train/IMG20230418125849_jpg.rf.ab8595e352fea59c944f45d3e2e0b67f.jpg")
imglst = []
root = r"C:\Users\LTSS2023\Documents\elhadjmb/datasets/train/"
for image in os.listdir(root):
    # check if the image ends with png
    if (image.endswith(".jpg")):
        imglst.append(root + image)
        random.shuffle(imglst)

imglst = list(map(lambda x: cv2.imread(x), imglst))

for img in imglst:
    # Run the pipeline
    json_stats, drawn_image = pipeline.run(img)
    pprint(json_stats)
    display(drawn_image)
    break
