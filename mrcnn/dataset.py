import json
import os
import random
import shutil
from pathlib import Path
from pprint import pprint

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from imgaug import augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
from matplotlib import pyplot as plt, patches
from pycocotools.coco import COCO
from shapely.geometry import Polygon as Pol
from shapely.ops import unary_union
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from mrcnn.utilities import msg, wrp


class CocoDataset(Dataset):
    JSON_FILE = r"/_annotations.coco.json"
    TRAIN_DIR = r"/train"
    BACKUP_FOLDER = r"/backup"
    AUG_DIR = r"/augmented"
    ANALYSIS_JSON='/analysis.json'

    @wrp
    def __init__(self, root, version=0, enhance=False):
        self.mean = None
        self.std = None
        self.anchor_size = None
        self.anchor_ratio = None
        self.root_dir = root
        self.backup_dir = str(Path(str(os.path.dirname(root)))) + self.BACKUP_FOLDER
        self.version = version
        self.images = self.root_dir + self.TRAIN_DIR
        self.annotations = self.images + self.JSON_FILE

        self.coco = COCO(self.annotations)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.img_ids = self.coco.getImgIds()

        self.sc_annotations = None
        self.enhance = enhance
        self.augmented_root = str(Path(str(os.path.dirname(root)))) + self.AUG_DIR
        self.augmented_images = self.augmented_root + self.TRAIN_DIR
        self.augmented_annotations = self.augmented_images + self.JSON_FILE

        self.categories = None
        self.num_classes = None

    def __getitem__(self, index):
        # Load image
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        masks = []
        labels = []

        for ann in anns:
            # Bounding box [x, y, width, height]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])

            # Masks (COCO polygon format)
            masks.append(self.coco.annToMask(ann))

            # Labels (assuming only one class)
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Convert list of numpy.ndarrays to a single numpy.ndarray, then to a tensor
        masks = np.array(masks)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Calculate the area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "masks": masks,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": area,
            "iscrowd": iscrowd,
        }

        img, target = self.apply_transform(img, target)
        return img, target

    def __len__(self):
        return len(self.ids)

    def apply_transform(self, img, target):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        img = transform(img)

        return img, target

    @wrp
    def run(self):
        self.analyze()
        self.backup()
        self.fetch_info()

        if self.enhance:
            self.enhance_images()

        # self.visualize_annotations(0)
        self.simplify_polygons()
        self.analyze()
        self.mean, self.std = self.calculate_mean_std()
        self.anchor_size, self.anchor_ratio = self.get_optimal_anchors()

    def fetch_info(self):
        with open(self.annotations) as f:
            data = json.load(f)

        self.categories = data['categories']
        self.num_classes = len(self.categories)

    @wrp
    def enhance_images(self):
        filenames = [os.path.join(self.images, f) for f in os.listdir(self.images) if
                     (f.endswith(".jpg") or f.endswith(".png"))]

        for filename in tqdm(filenames, desc="Processing images..."):
            self.process_and_save(filename)

    def process_and_save(self, filename):
        img = cv2.imread(filename)
        enhanced = self.process_image(img)
        enhanced_filename = filename  # os.path.splitext(filename)[0] + "_enhanced" + os.path.splitext(filename)[1]
        cv2.imwrite(enhanced_filename, enhanced)

    def process_image(self, image):
        image = self.increase_saturation(image)
        image = self.equalize_histogram(image)
        image = self.denoise(image)
        image = self.sharpen_image(image)

        return image

    def equalize_histogram(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    def denoise(self, img):
        return cv2.fastNlMeansDenoisingColored(img)

    def sharpen_image(self, img):
        blurred = cv2.GaussianBlur(img, (0, 0), 5)
        return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    def increase_saturation(self, img, saturation_scale=1.1):
        # Convert BGR image to HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float32')

        # Scale the saturation channel
        img_hsv[:, :, 1] = img_hsv[:, :, 1] * saturation_scale

        # Make sure the values stay in the valid range [0, 255]
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)

        # Convert the HSV image back to BGR
        img_bgr = cv2.cvtColor(img_hsv.astype('uint8'), cv2.COLOR_HSV2BGR)
        return img_bgr

    @wrp
    def backup(self):
        if os.path.exists(self.backup_dir):
            return  # shutil.rmtree(destination_folder)
        # Copy the folder and its contents
        shutil.copytree(self.root_dir, self.backup_dir, dirs_exist_ok=True)

    @wrp
    def analyze(self, augmented=False):
        annotations_file = self.annotations if not augmented else self.augmented_annotations

        with open(annotations_file, 'r') as f:
            data = json.load(f)

        analysis = {}

        # Basic Counts
        analysis['num_images'] = len(data['images'])
        analysis['num_annotations'] = len(data['annotations'])
        analysis['num_classes'] = len(data['categories'])

        # Initialize
        instances_per_class = {}
        images_with_annotations = set()
        classes_with_annotations = set()
        annotations_per_image = {}
        bbox_sizes = []
        bbox_aspect_ratios = []
        bbox_heights = []
        bbox_widths = []
        total_vertices = 0
        polygon_areas = []
        num_vertices = []
        polygon_aspect_ratios = []

        for annotation in data['annotations']:
            image_id = annotation['image_id']
            category_id = annotation['category_id']
            bbox = annotation['bbox']
            segmentation = annotation['segmentation'][0]

            # Instances per class
            instances_per_class[category_id] = instances_per_class.get(category_id, 0) + 1

            # Images and classes with annotations
            images_with_annotations.add(image_id)
            classes_with_annotations.add(category_id)

            # Annotations per image
            annotations_per_image[image_id] = annotations_per_image.get(image_id, 0) + 1

            # Bounding box statistics
            bbox_area = bbox[2] * bbox[3]
            bbox_sizes.append(bbox_area)
            bbox_aspect_ratios.append(bbox[2] / bbox[3])
            bbox_heights.append(bbox[3])
            bbox_widths.append(bbox[2])

            # Polygon statistics
            segmentation = annotation['segmentation'][0]
            segmentation = [(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)]
            polygon = Pol(segmentation)
            polygon_areas.append(polygon.area)
            num_vertices.append(len(segmentation) // 2)
            total_vertices += len(segmentation) // 2
            minx, miny, maxx, maxy = polygon.bounds
            polygon_aspect_ratios.append((maxx - minx) / (maxy - miny))

        # More statistics
        analysis['instances_per_class'] = instances_per_class
        analysis['num_images_with_annotations'] = len(images_with_annotations)
        analysis['num_classes_with_annotations'] = len(classes_with_annotations)
        analysis['annotations_per_image'] = {
            'min': np.min(list(annotations_per_image.values())),
            'max': np.max(list(annotations_per_image.values())),
            'avg': np.mean(list(annotations_per_image.values())),
            'median': np.median(list(annotations_per_image.values())),
            'std': np.std(list(annotations_per_image.values()))
        }
        analysis['bbox_stats'] = {
            'sizes': {'min': np.min(bbox_sizes), 'max': np.max(bbox_sizes), 'avg': np.mean(bbox_sizes),
                      'median': np.median(bbox_sizes), 'std': np.std(bbox_sizes)},
            'aspect_ratios': {'min': np.min(bbox_aspect_ratios), 'max': np.max(bbox_aspect_ratios),
                              'avg': np.mean(bbox_aspect_ratios), 'median': np.median(bbox_aspect_ratios),
                              'std': np.std(bbox_aspect_ratios)},
            'heights': {'min': np.min(bbox_heights), 'max': np.max(bbox_heights), 'avg': np.mean(bbox_heights),
                        'median': np.median(bbox_heights), 'std': np.std(bbox_heights)},
            'widths': {'min': np.min(bbox_widths), 'max': np.max(bbox_widths), 'avg': np.mean(bbox_widths),
                       'median': np.median(bbox_widths), 'std': np.std(bbox_widths)}
        }
        analysis['polygon_stats'] = {
            'total_vertices': total_vertices,
            'areas': {'min': np.min(polygon_areas), 'max': np.max(polygon_areas), 'avg': np.mean(polygon_areas),
                      'median': np.median(polygon_areas), 'std': np.std(polygon_areas)},
            'num_vertices': {'min': np.min(num_vertices), 'max': np.max(num_vertices), 'avg': np.mean(num_vertices),
                             'median': np.median(num_vertices), 'std': np.std(num_vertices)},
            'aspect_ratios': {'min': np.min(polygon_aspect_ratios), 'max': np.max(polygon_aspect_ratios),
                              'avg': np.mean(polygon_aspect_ratios), 'median': np.median(polygon_aspect_ratios),
                              'std': np.std(polygon_aspect_ratios)}
        }

        pprint(analysis)
        # Save analysis to JSON file
        analysis_file = (self.root_dir if not augmented else self.augmented_root) + self.ANALYSIS_JSON
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=4, default=str)

        return analysis

    @wrp
    def make_single_class(self):
        pass  # TODO: make this

    @wrp
    def get_optimal_anchors(self, num_sizes=5, num_ratios=3):
        with open(self.annotations, 'r') as file:
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

        msg(f"Optimal anchor sizes: {optimal_sizes}")
        msg(f"Optimal anchor ratios: {optimal_ratios}")

        return optimal_sizes, optimal_ratios

    @wrp
    def calculate_mean_std(self):
        image_files_dir = Path(self.images)

        files = list(image_files_dir.rglob('*.jpg'))

        mean = np.array([0., 0., 0.])
        std_temp = np.array([0., 0., 0.])

        num_samples = len(files)

        for i in tqdm(range(num_samples)):
            im = cv2.imread(str(files[i]))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(float) / 255.

            for j in range(3):
                mean[j] += np.mean(im[:, :, j])

        mean = list(mean / num_samples)
        msg(f"MEAN: {mean}")
        for i in tqdm(range(num_samples)):
            im = cv2.imread(str(files[i]))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(float) / 255.
            for j in range(3):
                std_temp[j] += ((im[:, :, j] - mean[j]) ** 2).sum() / \
                               (im.shape[0] * im.shape[1])

        std = list(np.sqrt(std_temp / num_samples))
        msg(f"STD: {std}")
        return mean, std

    @wrp
    def simplify_polygons(self, tolerance=0.005):
        """
            Simplify polygons in a COCO-format JSON annotation file using the Ramer-Douglas-Peucker algorithm.

            Parameters:
                tolerance (float): Tolerance parameter for the Ramer-Douglas-Peucker algorithm.
                Higher values result in more simplified polygons.
            """

        # Load the COCO annotations
        with open(self.annotations, 'r') as f:
            coco_data = json.load(f)

        # Iterate through each annotation and simplify the polygon
        for annotation in tqdm(coco_data['annotations'], desc='Simplifying polygons...'):
            original_polygon = annotation['segmentation'][0]
            original_polygon = [(original_polygon[i], original_polygon[i + 1]) for i in
                                range(0, len(original_polygon), 2)]

            # Create a Shapely Polygon object
            poly = Pol(original_polygon)

            # Simplify the polygon
            simplified_poly = poly.simplify(tolerance, preserve_topology=False)

            # Convert back to COCO format
            simplified_poly = list(unary_union(simplified_poly).exterior.coords)
            simplified_poly_coco = [coord for point in simplified_poly for coord in point]

            # Update the annotation
            annotation['segmentation'] = [simplified_poly_coco]

        # Save the modified annotations to a JSON file
        with open(self.annotations, 'w') as f:
            json.dump(coco_data, f)

    @wrp
    def old_visualize_annotations(self, num_images, augmented=False):
        """
        Visualize annotations on images.

        Parameters:
            num_images (int): Number of images to visualize. If 0, pick one image at random.
        """

        annotations_file = self.annotations if not augmented else self.augmented_annotations
        images_dir = self.images if not augmented else self.augmented_images
        # Load the COCO annotations
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        # Create a mapping from image ID to annotations and file names
        image_annotations = {}
        image_file_names = {}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(annotation)

        for image in coco_data['images']:
            image_file_names[image['id']] = image['file_name']

        # Select images to visualize
        if num_images == 0:
            selected_images = [random.choice(list(image_annotations.keys()))]
        else:
            selected_images = list(image_annotations.keys())[:num_images]

        # Visualize annotations
        for image_id in selected_images:
            image_path = f"{images_dir}/{image_file_names[image_id]}"
            image = Image.open(image_path)

            fig, ax = plt.subplots(1)
            ax.imshow(image)

            for annotation in image_annotations[image_id]:
                # Draw polygon
                polygon = annotation['segmentation'][0]
                polygon = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
                poly_patch = patches.Polygon(polygon, closed=True, edgecolor='r', facecolor='none')
                ax.add_patch(poly_patch)

                # Draw bounding box
                x, y, w, h = annotation['bbox']
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect)

                # Add label
                category_id = annotation['category_id']
                category_name = [cat['name'] for cat in coco_data['categories'] if cat['id'] == category_id][0]
                plt.text(x, y, category_name, color='white')

            plt.show()

    @wrp
    def visualize_annotations(self, num_images, augmented=False):
        """
        Visualize annotations on images.

        Parameters:
            num_images (int): Number of images to visualize. If 0, pick one image at random.
        """

        annotations_file = self.annotations if not augmented else self.augmented_annotations
        images_dir = self.images if not augmented else self.augmented_images
        # Load the COCO annotations
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        # Create a mapping from image ID to annotations and file names
        image_annotations = {}
        image_file_names = {}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(annotation)

        for image in coco_data['images']:
            image_file_names[image['id']] = image['file_name']

        # Select images to visualize
        if num_images == 0:
            selected_images = [random.choice(list(image_annotations.keys()))]
        else:
            selected_images = random.sample(list(image_annotations.keys()), min(num_images, len(image_annotations)))

        # Print some image_ids from annotations
        sample_annotation_ids = [ann['image_id'] for ann in coco_data['annotations'][:10]]
        print("Sample annotation image_ids:", sample_annotation_ids)

        # Print some ids from images
        sample_image_ids = [img['id'] for img in coco_data['images'][:10]]
        print("Sample image ids:", sample_image_ids)

        # Check for mismatches
        mismatched_ids = set(sample_annotation_ids) - set(sample_image_ids)
        print("Mismatched ids:", mismatched_ids)

        # Visualize annotations
        for image_id in selected_images:
            image_file_name = [img['file_name'] for img in coco_data['images'] if img['id'] == image_id][0]
            image_path = f"{images_dir}/{image_file_name}"
            image = Image.open(image_path)

            fig, ax = plt.subplots(1)
            ax.imshow(image)

            for annotation in image_annotations[image_id]:
                # Draw polygon
                polygon = annotation['segmentation'][0]
                polygon = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
                poly_patch = patches.Polygon(polygon, closed=True, edgecolor='r', facecolor='none')
                ax.add_patch(poly_patch)

                # Draw bounding box
                x, y, w, h = annotation['bbox']
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect)

                # Add label
                category_id = annotation['category_id']
                category_name = [cat['name'] for cat in coco_data['categories'] if cat['id'] == category_id][0]
                plt.text(x, y, category_name, color='white')

            plt.show()

    @wrp
    def old_augment_dataset(self, annotations_file, images_folder, output_folder, target_count, simple):
        # Initialize COCO object
        coco = COCO(annotations_file)

        # Create output directories
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "annotations"), exist_ok=True)

        # Load categories and initialize counts
        categories = coco.loadCats(coco.getCatIds())
        category_counts = {cat['id']: 0 for cat in categories}

        # Define the categories
        color_augmenters = [
            iaa.Add((-2, 2)),
            iaa.Multiply((0.9, 1.1)),
            iaa.AddToHueAndSaturation((-3, 3)),
            iaa.GammaContrast((0.9, 1.1), per_channel=True)
        ]

        noise_augmenters = [
            iaa.JpegCompression((70, 90)),
            iaa.SaltAndPepper((0.01, 0.1)),
            iaa.CoarseDropout(0.01, size_percent=(0.01, 0.1), per_channel=True),
            iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(0.5))
        ]

        blur_augmenters = [
            iaa.MotionBlur((3, 7)),
            iaa.GaussianBlur((0.5, 1.0))
        ]

        edge_augmenters = [
            iaa.Sharpen((0.5, 0.8)),
            iaa.Emboss((0.1, 0.6)),
            iaa.EdgeDetect((0.1, 0.5))
        ]

        flip_augmenters = [
            iaa.Fliplr(0.6),
            iaa.Flipud(0.6),
            iaa.geometric.Affine(rotate=(-90, 90), shear=(10, 10), mode='edge')
        ]

        weather_augmenters = [
            iaa.Clouds(),
            iaa.Fog()
        ]

        # Create the augmenter
        augmenter = iaa.Sequential([
            iaa.Sometimes(np.random.uniform(0.4, 0.7), iaa.OneOf(color_augmenters)),
            iaa.Sometimes(np.random.uniform(0, 0.7), iaa.OneOf(noise_augmenters)),
            iaa.Sometimes(np.random.uniform(0.2, 0.5), iaa.OneOf(blur_augmenters)),
            iaa.Sometimes(np.random.uniform(0.4, 0.7), iaa.OneOf(edge_augmenters)),
            iaa.Sometimes(np.random.uniform(0.8, 1), iaa.SomeOf((1, 2), flip_augmenters)),
            iaa.Sometimes(np.random.uniform(0, 0.8), iaa.OneOf(weather_augmenters)),
        ], random_order=True) if not simple else iaa.Sequential([
            # iaa.Sometimes(np.random.uniform(0.4, 0.7), iaa.SomeOf((1, 2), edge_augmenters)),
            iaa.Sometimes(1, iaa.SomeOf((1, 2), flip_augmenters)),
            # iaa.Sometimes(np.random.uniform(0, 0.2), iaa.OneOf(color_augmenters)),
        ], random_order=True)

        # Initialize new annotations
        new_annotations = {'images': [], 'annotations': [], 'categories': categories}
        annotation_id = 0

        # Loop through each image
        for img_id in coco.imgs.keys():
            img_data = coco.loadImgs([img_id])[0]
            img = cv2.imread(os.path.join(images_folder, img_data['file_name']))

            # Loop through each annotation in the image
            for ann_id in coco.getAnnIds(imgIds=[img_id]):
                ann_data = coco.loadAnns([ann_id])[0]
                category_counts[ann_data['category_id']] += 1

                # Check if augmentation is needed for this category
                if category_counts[ann_data['category_id']] < target_count:
                    # Perform augmentation
                    keypoints = ann_data['segmentation'][0]
                    keypoints = [Keypoint(x=keypoints[i], y=keypoints[i + 1]) for i in range(0, len(keypoints), 2)]
                    keypoints_aug = augmenter.augment_keypoints([keypoints])[0]

                    # Update image and annotation
                    img_aug = augmenter.augment_image(img)
                    ann_data['segmentation'][0] = [coord for kp in keypoints_aug for coord in [kp.x, kp.y]]
                    ann_data['id'] = annotation_id
                    annotation_id += 1

                    # Save augmented image and update annotations
                    new_img_file = f"{img_data['id']}_aug.jpg"
                    cv2.imwrite(os.path.join(output_folder, "images", new_img_file), img_aug)
                    new_annotations['images'].append({'id': img_data['id'], 'file_name': new_img_file})
                    new_annotations['annotations'].append(ann_data)

        # Save new annotations
        with open(os.path.join(output_folder, "annotations", "annotations_augmented.json"), 'w') as f:
            json.dump(new_annotations, f)

    @wrp
    def keypoint_augment_dataset(self, target_count, simple):
        # Initialize COCO object
        coco = COCO(self.annotations)

        # Create output directories
        os.makedirs(self.augmented_root, exist_ok=True)
        os.makedirs(self.augmented_images, exist_ok=True)

        # Load categories and initialize counts
        categories = coco.loadCats(coco.getCatIds())
        category_counts = {cat['id']: 0 for cat in categories}

        # Define the categories
        color_augmenters = [
            iaa.Add((-2, 2)),
            iaa.Multiply((0.9, 1.1)),
            iaa.AddToHueAndSaturation((-3, 3)),
            iaa.GammaContrast((0.9, 1.1), per_channel=True)
        ]

        noise_augmenters = [
            iaa.JpegCompression((70, 90)),
            iaa.SaltAndPepper((0.01, 0.1)),
            iaa.CoarseDropout(0.01, size_percent=(0.01, 0.1), per_channel=True),
            iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(0.5))
        ]

        blur_augmenters = [
            iaa.MotionBlur((3, 7)),
            iaa.GaussianBlur((0.5, 1.0))
        ]

        edge_augmenters = [
            iaa.Sharpen((0.5, 0.8)),
            iaa.Emboss((0.1, 0.6)),
            iaa.EdgeDetect((0.1, 0.5))
        ]

        flip_augmenters = [
            iaa.Fliplr(0.6),
            iaa.Flipud(0.6),
            iaa.geometric.Affine(rotate=(-90, 90), shear=(10, 10), mode='edge')
        ]

        weather_augmenters = [
            iaa.Clouds(),
            iaa.Fog()
        ]

        # Create the augmenter
        augmenter = iaa.Sequential([
            iaa.Sometimes(np.random.uniform(0.4, 0.7), iaa.OneOf(color_augmenters)),
            iaa.Sometimes(np.random.uniform(0, 0.7), iaa.OneOf(noise_augmenters)),
            iaa.Sometimes(np.random.uniform(0.2, 0.5), iaa.OneOf(blur_augmenters)),
            iaa.Sometimes(np.random.uniform(0.4, 0.7), iaa.OneOf(edge_augmenters)),
            iaa.Sometimes(np.random.uniform(0, 0.8), iaa.OneOf(weather_augmenters)),
        ], random_order=True) if not simple else iaa.Sequential([
            # iaa.Sometimes(np.random.uniform(0.4, 0.7), iaa.SomeOf((1, 2), edge_augmenters)),
            iaa.Sometimes(1, iaa.SomeOf((1, 2), flip_augmenters)),
            # iaa.Sometimes(np.random.uniform(0, 0.2), iaa.OneOf(color_augmenters)),
        ], random_order=True)

        # Initialize new annotations
        new_annotations = {'images': [], 'annotations': [], 'categories': categories}
        annotation_id = 0

        # Loop through each image
        for img_id in tqdm(coco.imgs.keys(), desc="Augmenting images..."):
            img_data = coco.loadImgs([img_id])[0]
            img = cv2.imread(os.path.join(self.images, img_data['file_name']))

            # Loop through each annotation in the image
            for ann_id in coco.getAnnIds(imgIds=[img_id]):
                ann_data = coco.loadAnns([ann_id])[0]
                category_counts[ann_data['category_id']] += 1

                # Check if augmentation is needed for this category
                if category_counts[ann_data['category_id']] < target_count:
                    # Perform augmentation
                    keypoints = ann_data['segmentation'][0]
                    keypoints = [Keypoint(x=keypoints[i], y=keypoints[i + 1]) for i in range(0, len(keypoints), 2)]
                    keypoints_on_image = KeypointsOnImage(keypoints, shape=img.shape)

                    # Augment keypoints
                    img_aug, keypoints_aug = augmenter(image=img, keypoints=keypoints_on_image)

                    # Extract transformed keypoints
                    transformed_keypoints = [(kp.x, kp.y) for kp in keypoints_aug.keypoints]

                    # Recalculate bounding box
                    min_x = min([kp[0] for kp in transformed_keypoints])
                    min_y = min([kp[1] for kp in transformed_keypoints])
                    max_x = max([kp[0] for kp in transformed_keypoints])
                    max_y = max([kp[1] for kp in transformed_keypoints])
                    bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

                    # Update image and annotation
                    ann_data['segmentation'][0] = [float(coord) for kp in transformed_keypoints for coord in kp]
                    ann_data['bbox'] = [float(coord) for coord in bbox]
                    ann_data['id'] = annotation_id
                    annotation_id += 1

                    # Save augmented image and update annotations
                    new_img_file = f"{img_data['id']}_aug.jpg"
                    cv2.imwrite(os.path.join(self.augmented_images, new_img_file), img_aug)
                    new_annotations['images'].append({'id': img_data['id'], 'file_name': new_img_file})
                    new_annotations['annotations'].append(ann_data)

        # Save new annotations
        with open(os.path.join(self.augmented_annotations), 'w') as f:
            json.dump(new_annotations, f)

    @wrp
    def augment_dataset(self, target_count, simple):
        # Initialize COCO object
        coco = COCO(self.annotations)

        # Create output directories
        os.makedirs(self.augmented_root, exist_ok=True)
        os.makedirs(self.augmented_images, exist_ok=True)

        # Load categories and initialize counts
        categories = coco.loadCats(coco.getCatIds())
        category_counts = {cat['id']: 0 for cat in categories}

        # Define the categories
        color_augmenters = [
            iaa.Add((-2, 2)),
            iaa.Multiply((0.9, 1.1)),
            iaa.AddToHueAndSaturation((-3, 3)),
            iaa.GammaContrast((0.9, 1.1), per_channel=True)
        ]

        noise_augmenters = [
            iaa.JpegCompression((70, 90)),
            iaa.SaltAndPepper((0.01, 0.1)),
            iaa.CoarseDropout(0.01, size_percent=(0.01, 0.1), per_channel=True),
            iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(0.5))
        ]

        blur_augmenters = [
            iaa.MotionBlur((3, 7)),
            iaa.GaussianBlur((0.5, 1.0))
        ]

        edge_augmenters = [
            iaa.Sharpen((0.5, 0.8)),
            iaa.Emboss((0.1, 0.6)),
            iaa.EdgeDetect((0.1, 0.5))
        ]

        flip_augmenters = [
            iaa.Fliplr(0.6),
            iaa.Flipud(0.6),
            iaa.geometric.Affine(rotate=(-90, 90), shear=(10, 10), mode='edge')
        ]

        weather_augmenters = [
            iaa.Clouds(),
            iaa.Fog()
        ]

        # Create the augmenter
        augmenter = iaa.Sequential([
            iaa.Sometimes(np.random.uniform(0.4, 0.7), iaa.OneOf(color_augmenters)),
            iaa.Sometimes(np.random.uniform(0, 0.7), iaa.OneOf(noise_augmenters)),
            iaa.Sometimes(np.random.uniform(0.2, 0.5), iaa.OneOf(blur_augmenters)),
            iaa.Sometimes(np.random.uniform(0.4, 0.7), iaa.OneOf(edge_augmenters)),
            # iaa.Sometimes(np.random.uniform(0.8, 1), iaa.SomeOf((1, 2), flip_augmenters)),
            iaa.Sometimes(np.random.uniform(0, 0.8), iaa.OneOf(weather_augmenters)),
        ], random_order=True) if not simple else iaa.Sequential([
            # iaa.Sometimes(np.random.uniform(0.4, 0.7), iaa.SomeOf((1, 2), edge_augmenters)),
            iaa.Sometimes(1, iaa.SomeOf((1, 2), flip_augmenters)),
            # iaa.Sometimes(np.random.uniform(0, 0.2), iaa.OneOf(color_augmenters)),
        ], random_order=True)

        # Initialize new annotations
        new_annotations = {'images': [], 'annotations': [], 'categories': categories, 'info': coco.dataset['info'],
                           'licenses': coco.dataset['licenses']}
        annotation_id = 0  # Initialize annotation ID counter
        new_image_id = max(coco.imgs.keys()) * 1000  # Start new image IDs using the method discussed

        # Loop through each image
        for img_id in tqdm(coco.imgs.keys(), desc="Augmenting images..."):
            img_data = coco.loadImgs([img_id])[0]
            img = cv2.imread(os.path.join(self.images, img_data['file_name']))

            # Loop through each annotation in the image
            for ann_id in coco.getAnnIds(imgIds=[img_id]):
                ann_data = coco.loadAnns([ann_id])[0]
                category_counts[ann_data['category_id']] += 1

                # Check if augmentation is needed for this category
                while category_counts[ann_data['category_id']] < target_count:
                    # Perform augmentation
                    keypoints = ann_data['segmentation'][0]
                    polygon_points = [(keypoints[i], keypoints[i + 1]) for i in range(0, len(keypoints), 2)]
                    polygon = Polygon(polygon_points)
                    polygons_on_image = PolygonsOnImage([polygon], shape=img.shape)

                    # Augment polygons
                    img_aug, polygons_aug = augmenter(image=img, polygons=polygons_on_image)

                    # Extract transformed polygons
                    transformed_polygon = polygons_aug.polygons[0]
                    transformed_keypoints = transformed_polygon.exterior

                    # Recalculate bounding box
                    min_x = min([kp[0] for kp in transformed_keypoints])
                    min_y = min([kp[1] for kp in transformed_keypoints])
                    max_x = max([kp[0] for kp in transformed_keypoints])
                    max_y = max([kp[1] for kp in transformed_keypoints])
                    bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

                    # Check the area of the bounding box
                    original_area = ann_data['bbox'][2] * ann_data['bbox'][3]
                    new_area = (max_x - min_x) * (max_y - min_y)
                    if new_area < 0.1 * original_area or new_area > 2 * original_area:
                        continue  # Skip this augmentation

                    # Update image and annotation IDs
                    ann_data['image_id'] = new_image_id
                    ann_data['id'] = annotation_id
                    annotation_id += 1  # Increment annotation ID

                    # Save augmented image and update annotations
                    new_img_file = f"{new_image_id}_aug.jpg"
                    cv2.imwrite(os.path.join(self.augmented_images, new_img_file), img_aug)
                    new_annotations['images'].append({'id': new_image_id, 'file_name': new_img_file})
                    new_annotations['annotations'].append(ann_data.copy())  # Use a copy to keep original intact

                    # Increment the count for this category
                    category_counts[ann_data['category_id']] += 1

                    # Increment new_image_id for next augmented image
                    new_image_id += 1

                    # Save new annotations
                with open(os.path.join(self.augmented_annotations), 'w') as f:
                    json.dump(new_annotations, f, indent=4)
