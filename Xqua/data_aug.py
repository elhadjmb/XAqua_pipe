import copy
import json
import math
import os
import random
import re
import shutil
import statistics
import uuid
from collections import Counter
from collections import defaultdict
from multiprocessing import Pool, freeze_support
from pathlib import Path
from pprint import pprint
from random import choices
from random import sample
from statistics import mean, stdev, median
from typing import List
from typing import List
import json
from rtree import index
from shapely.geometry import box
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from joblib import Parallel, delayed
from matplotlib import ticker
from rtree import index
from scipy.stats import gaussian_kde
from scipy.stats import iqr
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm
from tqdm import tqdm as q
from tqdm.auto import tqdm

# from tr7_clb_bkp import msg, wrp
import cv2
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm

from tr8_clb import wrp, msg

BAD_IMGS = [
    r"IMG20230418125507_001_jpg.rf.17d5829a9cc38db9c7ac26618fcb0b5d",
    r"IMG20230418125507_001_jpg.rf.c35e080bf89ee1b9aaeb9f0f68e8a966",
    r"IMG20230418125507_001_jpg.rf.c68b9a83175b7b8de42bc213dfc86a56",
    r"IMG20230418125507_002_jpg.rf.5ef0e57dcf1c6c531feb48fd25d11dae",
    r"IMG20230418125507_002_jpg.rf.53910f331b5996d25e960a52707e6fbc",
    r"IMG20230418125507_002_jpg.rf.537794e5b232c84fc1b9bab181523d22",
    r"IMG20230418125507_004_jpg.rf.3b9954965ae7bc6b1fa5edaacd73c3e2",
    r"IMG20230418125507_004_jpg.rf.a07d3f1c95a0a9e44542f1e5d829442c",
    r"IMG20230418125507_004_jpg.rf.f8cbd2cc874b24a1a525400acb7cfc24",
    r"IMG20230418125507_001_jpg.rf.c35e080bf89ee1b9aaeb9f0f68e8a966",
    r"IMG20230418125507_001_jpg.rf.c35e080bf89ee1b9aaeb9f0f68e8a966",
    r"IMG20230418125507_001_jpg.rf.17d5829a9cc38db9c7ac26618fcb0b5d",
    r"IMG20230418125507_001_jpg.rf.c68b9a83175b7b8de42bc213dfc86a56",
    r"IMG20230418125507_002_jpg.rf.5ef0e57dcf1c6c531feb48fd25d11dae",
    r"IMG20230418125507_002_jpg.rf.53910f331b5996d25e960a52707e6fbc",
    r"IMG20230418125507_002_jpg.rf.537794e5b232c84fc1b9bab181523d22",
    r"IMG20230418125507_004_jpg.rf.3b9954965ae7bc6b1fa5edaacd73c3e2",
    r"IMG20230418125507_004_jpg.rf.f8cbd2cc874b24a1a525400acb7cfc24",
    r"IMG20230418125507_004_jpg.rf.a07d3f1c95a0a9e44542f1e5d829442c",
    r"IMG20230418125515_002_jpg.rf.0c0edfc2cbb7929252163b94a92e59a1",
    r"IMG20230418125515_002_jpg.rf.7c0b6c5f5049f1282cfc3a300c66ddd7",
    r"IMG20230418125515_002_jpg.rf.b03b4308b59ba298345f3dcc4a61272d",
    r"IMG20230418125515_003_jpg.rf.d57d42d8606fc9bfb4acc26e44c39a3d",
    r"IMG20230418125515_003_jpg.rf.da5d95a8082ef360149223279651d456",
    r"IMG20230418125512_001_jpg.rf.df51b90e7bcd0b1c0b897ed6658e32d1",
    r"IMG20230418125512_001_jpg.rf.1517986d39dba87c24bc1101c88d09c6",
    r"IMG20230418125512_001_jpg.rf.1aa25090a26ed4e72dc4db4a13c8c0d6",
    r"IMG20230418125515_003_jpg.rf.f6540b1b937bc4829a44cceeed475000",
    r"IMG20230418125512_004_jpg.rf.822b40df5db343371dffc34739ae601d",
    r"IMG20230418125512_004_jpg.rf.1995ba62f1dbdc454bc46a2b3708d54e",
    r"IMG20230418125512_004_jpg.rf.b13ccd63e33d987314fdc3b6a8252b92",
]


@wrp
def old_split_dataset(annotation_path, img_dir, output_dir, F):
    print(
        f"Running split_dataset with:\nannotation_path: {annotation_path}\nimg_dir: {img_dir}\noutput_dir: {output_dir}\nF: {F}")

    # Read the annotations
    with open(annotation_path) as f:
        annotations = json.load(f)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Determine the horizontal and vertical split count based on F (assuming F is a square number)
    split_count = int(math.sqrt(F))
    print(f"Each image will be split into {split_count}x{split_count} segments.")

    new_annotations = dict()
    new_images_info = []
    new_annotations_list = []

    # Iterate over the images in the dataset
    for image_info in annotations['images']:
        image_id = image_info['id']
        file_name = image_info['file_name']
        width = image_info['width']
        height = image_info['height']

        print(f"Processing image {image_id} ({file_name}) of size {width}x{height}...")

        # Read the image
        img = cv2.imread(img_dir + '/' + file_name)

        # Calculate the width and height of the split images
        new_width = width // split_count
        new_height = height // split_count

        # Split the image and adjust the annotations
        new_images = []
        for i in range(split_count):
            for j in range(split_count):
                # Extract the image segment
                new_image = img[i * new_height:(i + 1) * new_height, j * new_width:(j + 1) * new_width]
                new_images.append(new_image)

                print(f"Created new image segment {len(new_images)} of size {new_width}x{new_height}.")

                # Adjust the annotations
                for annotation in annotations['annotations']:
                    if annotation['image_id'] == image_id:
                        x, y, w, h = annotation['bbox']
                        if j * new_width <= x < (j + 1) * new_width and i * new_height <= y < (i + 1) * new_height:
                            # Check if the bounding box is within the new image segment
                            if j * new_width <= x + w <= (j + 1) * new_width and i * new_height <= y + h <= (
                                    i + 1) * new_height:
                                new_annotation = annotation.copy()
                                new_annotation['bbox'] = [x - j * new_width, y - i * new_height, w, h]
                                new_annotation['image_id'] = len(new_images_info) + len(new_images) - 1
                                new_annotation['id'] = len(new_annotations_list)  # Update the annotation ID here
                                new_annotations_list.append(new_annotation)
                                # print(f"Adjusted annotation {annotation['id']} for new image segment. New annotation: {new_annotation}")

        # Save the new images and update the image info list
        for k, new_image in enumerate(new_images):
            new_file_name = file_name.replace('.jpg', f'_{k}.jpg')
            cv2.imwrite(output_dir + '/' + new_file_name, new_image)
            new_id = image_id * F + k
            new_images_info.append({'id': new_id, 'file_name': new_file_name, 'width': new_width, 'height': new_height})
            print(f"Saved new image segment {k + 1} as {new_file_name} and updated image information in annotations.")

    # After all images and annotations have been processed...
    image_ids = set([image['id'] for image in new_images_info])
    annotation_ids = set([annotation['id'] for annotation in
                          new_annotations_list])  # We're now checking annotation IDs, not annotation image IDs
    assert len(image_ids) == len(new_images_info), "Duplicate image ids detected!"
    assert len(annotation_ids) == len(new_annotations_list), "Duplicate annotation ids detected!"

    new_annotations = annotations
    new_annotations['images'] = new_images_info
    new_annotations['annotations'] = new_annotations_list

    # Write the new annotations
    with open(output_dir + '/annotations.json', 'w') as f:
        json.dump(new_annotations, f)

    print(
        f"Saved {len(new_annotations_list)} new annotations and the updated annotations to {output_dir}/annotations.json.")


@wrp
def split_dataset(annotation_path, img_dir, output_dir, number_of_imgs):
    print(
        f"Running split_dataset with:\nannotation_path: {annotation_path}\nimg_dir: {img_dir}\noutput_dir: {output_dir}\nF: {number_of_imgs}")

    # Read the annotations
    with open(annotation_path) as f:
        annotations = json.load(f)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Determine the horizontal and vertical split count based on F (assuming F is a square number)
    split_count = int(math.sqrt(number_of_imgs))
    print(f"Each image will be split into {split_count}x{split_count} segments.")

    new_annotations = dict()
    new_images_info = []
    new_annotations_list = []
    new_annotation_id = 0

    # Iterate over the images in the dataset
    for image_info in tqdm(annotations['images']):
        image_id = image_info['id']
        file_name = image_info['file_name']
        width = image_info['width']
        height = image_info['height']

        # Read the image
        img = cv2.imread(img_dir + '/' + file_name)

        # Calculate the width and height of the split images
        new_width = width // split_count
        new_height = height // split_count

        # Split the image and adjust the annotations
        new_images = []
        for i in range(split_count):
            for j in range(split_count):
                # Extract the image segment
                new_image = img[i * new_height:(i + 1) * new_height, j * new_width:(j + 1) * new_width]
                new_images.append(new_image)

                # Adjust the annotations
                for annotation in annotations['annotations']:
                    if annotation['image_id'] == image_id:
                        x, y, w, h = annotation['bbox']
                        # Check if the bounding box overlaps with the new image
                        if not (x > (j + 1) * new_width or x + w < j * new_width or y > (
                                i + 1) * new_height or y + h < i * new_height):
                            x_new = max(x - j * new_width, 0)
                            y_new = max(y - i * new_height, 0)
                            w_new = min(x + w - j * new_width, new_width) - x_new
                            h_new = min(y + h - i * new_height, new_height) - y_new

                            new_annotation = annotation.copy()
                            new_annotation['bbox'] = [x_new, y_new, w_new, h_new]
                            new_annotation['image_id'] = len(new_images_info) + len(new_images) - 1
                            new_annotation['id'] = new_annotation_id
                            new_annotations_list.append(new_annotation)
                            new_annotation_id += 1

        # Save the new images and update the image info list
        for k, new_image in enumerate(new_images):
            new_file_name = file_name.replace('.jpg', f'_{k}.jpg')
            cv2.imwrite(output_dir + '/' + new_file_name, new_image)
            new_id = image_id * number_of_imgs + k
            new_images_info.append({'id': new_id, 'file_name': new_file_name, 'width': new_width, 'height': new_height})

    # After all images and annotations have been processed...
    new_annotations = annotations
    new_annotations['images'] = new_images_info
    new_annotations['annotations'] = new_annotations_list

    # Write the new annotations
    with open(output_dir + '/annotations.json', 'w') as number_of_imgs:
        json.dump(new_annotations, number_of_imgs)

    print(
        f"Saved {len(new_annotations_list)} new annotations and the updated annotations to {output_dir}/annotations.json.")


@wrp
def split_and_overlap_dataset(annotation_path, img_dir, output_dir, number_of_imgs, stride_ratio=0.5):
    print(
        f"Running split_and_overlap_dataset with:\nannotation_path: {annotation_path}\nimg_dir: {img_dir}\noutput_dir: {output_dir}\nF: {number_of_imgs}\nstride_ratio: {stride_ratio}")

    # Read the annotations
    with open(annotation_path) as f:
        annotations = json.load(f)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Determine the horizontal and vertical split count based on F (assuming F is a square number)
    split_count = int(math.sqrt(number_of_imgs))
    print(f"Each image will be split into {split_count}x{split_count} segments.")

    # Calculate the stride size based on the stride_ratio
    stride_size = int(split_count * (1 - stride_ratio))

    new_annotations = dict()
    new_images_info = []
    new_annotations_list = []
    new_annotation_id = 0

    # Iterate over the images in the dataset
    for image_info in tqdm(annotations['images']):
        image_id = image_info['id']
        file_name = image_info['file_name']
        width = image_info['width']
        height = image_info['height']

        # Read the image
        img = cv2.imread(os.path.join(img_dir, file_name))

        # Calculate the width and height of the split images
        new_width = width // split_count
        new_height = height // split_count

        # Calculate the stride size based on the stride_ratio
        stride_size = int(new_width * (1 - stride_ratio))

        # Split the image and adjust the annotations
        tile_counter = 0
        for i in range(0, height - new_height + 1, stride_size):
            for j in range(0, width - new_width + 1, stride_size):
                # Extract the image segment
                new_image = img[i:i + new_height, j:j + new_width]

                # Save the new images and update the image info list
                new_file_name = file_name.replace('.jpg', f'_{tile_counter}.jpg')
                cv2.imwrite(os.path.join(output_dir, new_file_name), new_image)
                new_id = len(new_images_info)
                new_images_info.append(
                    {'id': new_id, 'file_name': new_file_name, 'width': new_width, 'height': new_height})

                # Adjust the annotations
                for annotation in annotations['annotations']:
                    if annotation['image_id'] == image_id:
                        x, y, w, h = annotation['bbox']
                        # Check if the bounding box overlaps with the new image
                        if not (x > (j + new_width) or x + w < j or y > (i + new_height) or y + h < i):
                            x_new = max(x - j, 0)
                            y_new = max(y - i, 0)
                            w_new = min(x + w - j, new_width) - x_new
                            h_new = min(y + h - i, new_height) - y_new

                            new_annotation = annotation.copy()
                            new_annotation['bbox'] = [x_new, y_new, w_new, h_new]
                            new_annotation['image_id'] = new_id
                            new_annotation['id'] = new_annotation_id
                            new_annotations_list.append(new_annotation)
                            new_annotation_id += 1
                tile_counter += 1

    # After all images and annotations have been processed...
    new_annotations = annotations
    new_annotations['images'] = new_images_info
    new_annotations['annotations'] = new_annotations_list

    # Write the new annotations
    with open(os.path.join(output_dir, 'annotations.json'), 'w') as number_of_imgs:
        json.dump(new_annotations, number_of_imgs)

    print(
        f"Saved {len(new_annotations_list)} new annotations and the updated annotations to {output_dir}/annotations.json.")


@wrp
def remove_edge_annotations(json_path: str, threshold: int = 5):
    """Removes annotations close to the edge of the image."""

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create a dictionary mapping image id to its size
    image_id_to_size = {image['id']: (image['width'], image['height']) for image in data['images']}

    new_annotations = []
    for annotation in tqdm(data['annotations']):
        bbox = annotation['bbox']  # [x,y,width,height]
        img_width, img_height = image_id_to_size[annotation['image_id']]

        # Check if the bounding box is too close to the edge
        if bbox[0] > threshold and bbox[1] > threshold and bbox[0] + bbox[2] < img_width - threshold and bbox[1] + bbox[
            3] < img_height - threshold:
            new_annotations.append(annotation)

    data['annotations'] = new_annotations

    with open(json_path, 'w') as f:
        json.dump(data, f)

@wrp
def overlap_split_dataset(annotation_path, img_dir, output_dir, number_of_imgs, stride_ratio=0.5):
    # Read the annotations
    with open(annotation_path) as f:
        annotations = json.load(f)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Determine the horizontal and vertical split count based on F (assuming F is a square number)
    split_count = int(math.sqrt(number_of_imgs))

    new_annotations = dict()
    new_images_info = []
    new_annotations_list = []
    new_annotation_id = 0

    # # Calculate the stride size based on the stride_ratio
    # stride_size = int(split_count * stride_ratio)

    # Sort the annotations
    annotations['annotations'].sort(key=lambda x: (x['image_id'], x['bbox'][1]))

    annotation_start_index = 0

    # Iterate over the images in the dataset
    for image_info in tqdm(annotations['images']):
        image_id = image_info['id']
        file_name = image_info['file_name']
        width = image_info['width']
        height = image_info['height']

        # Read the image
        img = cv2.imread(img_dir + '/' + file_name)

        # Calculate the width and height of the split images
        new_width = width // split_count
        new_height = height // split_count

        # Calculate the stride size based on the stride_ratio
        stride_size = int(new_width * (1 - stride_ratio))

        # Split the image and adjust the annotations
        new_images = []
        for i in range(0, height - new_height + 1, stride_size):
            for j in range(0, width - new_width + 1, stride_size):
                # Extract the image segment
                new_image = img[i:i + new_height, j:j + new_width]
                new_images.append(new_image)

                # Adjust the annotations
                for k in range(annotation_start_index, len(annotations['annotations'])):
                    annotation = annotations['annotations'][k]

                    if annotation['image_id'] == image_id and annotation['bbox'][1] < i + new_height:
                        x, y, w, h = annotation['bbox']

                        # Check if the bounding box is completely contained within the new image
                        if x >= j and x + w <= j + new_width and y >= i and y + h <= i + new_height:
                            x_new = x - j
                            y_new = y - i

                            new_annotation = annotation.copy()
                            new_annotation['bbox'] = [x_new, y_new, w, h]
                            new_annotation['image_id'] = len(new_images_info) + len(new_images) - 1
                            new_annotation['id'] = new_annotation_id
                            new_annotations_list.append(new_annotation)
                            new_annotation_id += 1

                        if annotation['bbox'][1] + annotation['bbox'][3] > i + new_height:
                            annotation_start_index = k
                            break
                    elif annotation['image_id'] != image_id or annotation['bbox'][1] >= i + new_height:
                        break

        # Save the new images and update the image info list
        for k, new_image in enumerate(new_images):
            new_file_name = file_name.replace('.jpg', f'_{k}.jpg')
            cv2.imwrite(output_dir + '/' + new_file_name, new_image)
            new_id = image_id * number_of_imgs + k
            new_images_info.append({'id': new_id, 'file_name': new_file_name, 'width': new_width, 'height': new_height})

    # After all images and annotations have been processed...
    new_annotations = annotations
    new_annotations['images'] = new_images_info
    new_annotations['annotations'] = new_annotations_list

    # Write the new annotations
    with open(output_dir + '/annotations.json', 'w') as number_of_imgs:
        json.dump(new_annotations, number_of_imgs)


@wrp
def xxsplit_dataset(annotation_path, img_dir, output_dir, number_of_imgs):
    print(
        f"Running split_dataset with:\nannotation_path: {annotation_path}\nimg_dir: {img_dir}\noutput_dir: {output_dir}\nF: {number_of_imgs}")

    # Read the annotations
    with open(annotation_path) as f:
        annotations = json.load(f)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Determine the horizontal and vertical split count based on F (assuming F is a square number)
    split_count = int(math.sqrt(number_of_imgs))
    print(f"Each image will be split into {split_count}x{split_count} segments.")

    new_annotations = dict()
    new_images_info = []
    new_annotations_list = []

    # Counter for new annotations and images
    new_annotation_id = len(annotations['annotations'])
    new_image_id = len(annotations['images'])

    # Iterate over the images in the dataset
    for image_info in tqdm(annotations['images']):
        image_id = image_info['id']
        file_name = image_info['file_name']
        width = image_info['width']
        height = image_info['height']

        # Read the image
        img = cv2.imread(img_dir + '/' + file_name)

        # Calculate the width and height of the split images
        new_width = width // split_count
        new_height = height // split_count

        # Split the image and adjust the annotations
        for i in range(split_count):
            for j in range(split_count):
                # Extract the image segment
                new_image = img[i * new_height:(i + 1) * new_height, j * new_width:(j + 1) * new_width]

                # Save the new image and update the image info list
                new_file_name = file_name.rsplit('.', 1)[0] + f'_{i}_{j}.jpg'
                cv2.imwrite(output_dir + '/' + new_file_name, new_image)

                new_images_info.append(
                    {'id': new_image_id, 'file_name': new_file_name, 'width': new_width, 'height': new_height})
                new_image_id += 1

                # Adjust the annotations
                for annotation in annotations['annotations']:
                    if annotation['image_id'] == image_id:
                        x, y, w, h = annotation['bbox']
                        # Check if the bounding box overlaps with the new image
                        if not (x > (j + 1) * new_width or x + w < j * new_width or y > (
                                i + 1) * new_height or y + h < i * new_height):
                            x_new = max(x - j * new_width, 0)
                            y_new = max(y - i * new_height, 0)
                            w_new = min(x + w - j * new_width, new_width) - x_new
                            h_new = min(y + h - i * new_height, new_height) - y_new

                            new_annotation = annotation.copy()
                            new_annotation['bbox'] = [x_new, y_new, w_new, h_new]
                            new_annotation['image_id'] = new_image_id - 1
                            new_annotation['id'] = new_annotation_id
                            new_annotations_list.append(new_annotation)
                            new_annotation_id += 1

    # After all images and annotations have been processed...
    new_annotations = annotations
    new_annotations['images'] = new_images_info
    new_annotations['annotations'] = new_annotations_list

    # Write the new annotations
    with open(output_dir + '/annotations.json', 'w') as number_of_imgs:
        json.dump(new_annotations, number_of_imgs)

    print(
        f"Saved {len(new_annotations_list)} new annotations and the updated annotations to {output_dir}/annotations.json.")


@wrp
def augment_dataset(images_dir, annotation_file, output_dir, num_augmented_images=25, simple=True):
    # Load the annotation file
    with open(annotation_file, 'r') as f:
        original_annotations = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

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

    augmented_annotations = {
        "images": [],
        "annotations": []
    }
    msg(f"Starting image augmentation to {num_augmented_images}...")
    # Iterate over the images in the dataset
    for image_info in tqdm(original_annotations['images']):
        # msg(f"Processing image {image_info['file_name']}...")
        image_id = image_info['id']
        image_file = os.path.join(images_dir, image_info['file_name'])
        image = Image.open(image_file)
        image_array = np.array(image)  # Convert image to numpy array

        # Find annotations for the current image
        annotations_for_image = [
            ann for ann in original_annotations['annotations'] if ann['image_id'] == image_id
        ]

        # Convert bounding box coordinates to BoundingBoxesOnImage format
        bounding_boxes = [
            BoundingBox(x1=ann['bbox'][0], y1=ann['bbox'][1], x2=ann['bbox'][0] + ann['bbox'][2],
                        y2=ann['bbox'][1] + ann['bbox'][3])
            for ann in annotations_for_image
        ]
        bbs = BoundingBoxesOnImage(bounding_boxes, shape=image_array.shape)

        # Calculate the area of the original bounding boxes
        original_areas = [bb.width * bb.height for bb in bounding_boxes]

        # Augment the image and annotations
        augmented_images = [image_array]
        augmented_bbs = [bbs]
        # -1 since we've already included the original
        for _ in range(num_augmented_images - 1):
            augmented_image, augmented_bbs_aug = augmenter(
                image=image_array, bounding_boxes=bbs)
            augmented_bbs_aug = augmented_bbs_aug.remove_out_of_image().clip_out_of_image()
            augmented_images.append(augmented_image)
            augmented_bbs.append(augmented_bbs_aug)

        # Check the sizes of the new bounding boxes
        for i, (augmented_image, augmented_bbs_aug) in enumerate(zip(augmented_images, augmented_bbs)):
            new_annotations = []
            for ann, bb, original_area in zip(annotations_for_image, augmented_bbs_aug.bounding_boxes,
                                              original_areas):
                new_area = bb.width * bb.height
                # If the new area is less than 10% or more than 200% of the original area, discard it
                if new_area < 0.1 * original_area or new_area > 2 * original_area:
                    continue

        # msg("Saving the augmented images and generating annotations...")
        for i, (augmented_image, augmented_bbs_aug) in enumerate(
                zip(augmented_images, augmented_bbs)):  # Exclude the original image
            new_image_id = image_id * 1000 + i
            # msg(f"Saving augmented image {new_image_id}...")
            new_image_file = os.path.join(output_dir, f"{new_image_id}.jpg")
            augmented_image = Image.fromarray(augmented_image.astype(
                np.uint8))  # Convert array back to image
            augmented_image.save(new_image_file)

            # msg(f"Generating annotations for augmented image {new_image_id + 1}...")
            new_annotations = []
            for ann, bb in zip(annotations_for_image, augmented_bbs_aug.bounding_boxes):
                new_ann = ann.copy()
                new_ann['id'] = len(
                    augmented_annotations['annotations']) + len(new_annotations)
                new_ann['image_id'] = new_image_id
                new_ann['bbox'] = [float(bb.x1), float(bb.y1), float(
                    bb.width), float(bb.height)]  # Update bounding box
                new_annotations.append(new_ann)

            augmented_annotations['images'].append({
                "id": new_image_id,
                "file_name": f"{new_image_id}.jpg"
            })
            augmented_annotations['annotations'].extend(new_annotations)

    augmented_annotations['info'] = original_annotations['info']
    augmented_annotations['licenses'] = original_annotations['licenses']
    augmented_annotations['categories'] = original_annotations['categories']
    # Save the augmented annotation file
    augmented_annotation_file = os.path.join(output_dir, 'annotations.json')
    msg("Saving augmented annotations...")
    with open(augmented_annotation_file, 'w') as f:
        json.dump(augmented_annotations, f, indent=4)


@wrp
def old_undersample_coco(input_json_path, images_dir, output_dir, output_json_path):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_json_path) as f:
        data = json.load(f)
    # Create a lookup dictionary for image filenames
    image_lookup = {image['id']: image['file_name'] for image in data['images']}

    annotations = data['annotations']
    images = data['images']
    categories = data['categories']

    msg("Counting number of instances per class")
    instances_per_class = Counter([anno['category_id']
                                   for anno in tqdm(annotations)])
    min_instances = min(instances_per_class.values())

    msg("Selecting min_instances from each class")
    # Create a dictionary that maps category IDs to their corresponding annotations
    annotations_by_category = defaultdict(list)
    for anno in annotations:
        annotations_by_category[anno['category_id']].append(anno)

    selected_annotations = []
    for cat_id in tqdm(instances_per_class.keys()):
        selected_annotations.extend(sample(annotations_by_category[cat_id], min_instances))

    # Convert selected_annotations to a set of IDs for faster membership checking
    selected_annotation_ids = set(anno['id'] for anno in selected_annotations)

    msg(
        "Updating the images list to include only images with selected annotations ... (this takes a lot of time please wait)")
    selected_images = [img for img in tqdm(images) if any(
        anno['image_id'] == img['id'] for anno in selected_annotations)]

    msg("Copying the selected images to the output directory")
    for img in tqdm(selected_images):
        shutil.copy(os.path.join(images_dir, img['file_name']), os.path.join(
            output_dir, img['file_name']))

    # # Create a dictionary to keep track of modified images
    # modified_images = {}
    #
    # # Sort annotations by image ID so that all annotations for the same image are processed consecutively
    # annotations.sort(key=lambda anno: anno['image_id'])
    #
    # # Iterate over all annotations
    # for annotation in tqdm(annotations):
    #     # Load the image if it hasn't been loaded yet
    #     image_id = annotation['image_id']
    #     if image_id not in modified_images:
    #         image_file = os.path.join(images_dir, image_lookup[image_id])
    #         modified_images[image_id] = cv2.imread(image_file)
    #
    #     # # If the annotation is not selected, draw over it
    #     # if annotation['id'] not in selected_annotation_ids:
    #     #     # Get the bounding box coordinates
    #     #     x, y, w, h = [int(coord) for coord in annotation['bbox']]
    #     #
    #     #     # Ensure the indices are within the image dimensions
    #     #     height, width, _ = modified_images[image_id].shape
    #     #     x = min(x, width - 1)
    #     #     y = min(y, height - 1)
    #     #     w = min(w, width - x)
    #     #     h = min(h, height - y)
    #     #
    #     #     # Get the border pixels of the bounding box
    #     #     top_border = modified_images[image_id][y, x:x + w]
    #     #     bottom_border = modified_images[image_id][min(y + h, height - 1), x:x + w]
    #     #     left_border = modified_images[image_id][y:y + h, x]
    #     #     right_border = modified_images[image_id][y:y + h, min(x + w, width - 1)]
    #     #
    #     #     # Calculate the average color of the border pixels
    #     #     border_pixels = np.concatenate((top_border, bottom_border, left_border, right_border), axis=0)
    #     #     avg_color = np.mean(border_pixels, axis=0)
    #     #
    #     #     # Draw over the bounding box with the average color
    #     #     cv2.rectangle(modified_images[image_id], (x, y), (x + w, y + h), avg_color, -1)
    #
    # # Save all modified images
    # for image_id, image in modified_images.items():
    #     output_file = os.path.join(output_dir, image_lookup[image_id])
    #     cv2.imwrite(output_file, image)

    # Update the annotations list to include only selected annotations
    annotations = [anno for anno in annotations if anno['id'] in selected_annotation_ids]

    annotation_dict = {'info': data['info'], 'licenses': data['licenses'],
                       'images': selected_images, 'annotations': annotations, 'categories': categories}
    msg("Saving the new annotations file")
    with open(output_json_path, 'w') as f:
        json.dump(annotation_dict, f)


@wrp
def undersample_coco(input_json_path, images_dir, output_dir, output_json_path):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_json_path) as f:
        data = json.load(f)
    # Create a lookup dictionary for image filenames
    image_lookup = {image['id']: image['file_name'] for image in data['images']}

    annotations = data['annotations']
    images = data['images']
    categories = data['categories']

    print("Counting number of instances per class")
    instances_per_class = Counter([anno['category_id'] for anno in tqdm(annotations)])

    # Calculate the median instances
    median_instances = statistics.median(instances_per_class.values())

    print("Selecting median_instances from each class")
    # Create a dictionary that maps category IDs to their corresponding annotations
    annotations_by_category = defaultdict(list)
    for anno in annotations:
        annotations_by_category[anno['category_id']].append(anno)

    selected_annotations = []
    for cat_id in tqdm(instances_per_class.keys()):
        # Ensure the category has at least median_instances
        if instances_per_class[cat_id] > median_instances:
            selected_annotations.extend(sample(annotations_by_category[cat_id], median_instances))
        else:
            selected_annotations.extend(annotations_by_category[cat_id])

    # Convert selected_annotations to a set of IDs for faster membership checking
    selected_annotation_ids = set(anno['id'] for anno in selected_annotations)

    print(
        "Updating the images list to include only images with selected annotations ... (this takes a lot of time please wait)")
    selected_images = [img for img in tqdm(images) if any(
        anno['image_id'] == img['id'] for anno in selected_annotations)]

    print("Copying the selected images to the output directory")
    for img in tqdm(selected_images):
        shutil.copy(os.path.join(images_dir, img['file_name']), os.path.join(
            output_dir, img['file_name']))

    # Update the annotations list to include only selected annotations
    annotations = [anno for anno in annotations if anno['id'] in selected_annotation_ids]

    annotation_dict = {'info': data['info'], 'licenses': data['licenses'],
                       'images': selected_images, 'annotations': annotations, 'categories': categories}
    print("Saving the new annotations file")
    with open(output_json_path, 'w') as f:
        json.dump(annotation_dict, f)

@wrp
def oversample_coco(input_json_path, images_dir, output_dir, output_json_path, threshold=0.2):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_json_path) as f:
        data = json.load(f)

    annotations = data['annotations']
    images = data['images']
    categories = data['categories']

    instances_per_class = Counter([anno['category_id'] for anno in annotations])

    # Calculate mean number of instances per category
    mean_instances = int(sum(instances_per_class.values()) / len(instances_per_class))

    msg("Duplicating instances for each class until within threshold")
    oversampled_annotations = []
    oversampled_images = []

    new_image_id = max([img['id'] for img in images]) + 1
    new_anno_id = max([anno['id'] for anno in annotations]) + 1

    added_images = set()
    added_annos = set()

    for cat_id, num_instances in tqdm(instances_per_class.items()):
        annos = [anno for anno in annotations if anno['category_id'] == cat_id]
        imgs = [img for img in images if any(anno['image_id'] == img['id'] for anno in annos)]

        target_instances = mean_instances

        while num_instances < target_instances:
            img = choices(imgs)[0]
            img_annos = [anno for anno in annos if anno['image_id'] == img['id']]

            copied_image = img.copy()
            copied_image['id'] = new_image_id
            oversampled_images.append(copied_image)

            for anno in img_annos:
                copied_anno = anno.copy()
                copied_anno['id'] = new_anno_id
                copied_anno['image_id'] = new_image_id

                # Decrement category_id by 1 if it's in valid category_ids
                # if copied_anno['category_id'] in valid_category_ids:
                #     copied_anno['category_id'] -= 1

                oversampled_annotations.append(copied_anno)
                new_anno_id += 1

            new_image_id += 1
            num_instances += 1

        # Add the original images and annotations to the oversampled list if they haven't already been added
        for img in tqdm(imgs):
            if img['id'] not in added_images:
                oversampled_images.append(img)
                added_images.add(img['id'])

        for anno in tqdm(annos):
            if anno['id'] not in added_annos:
                oversampled_annotations.append(anno)
                added_annos.add(anno['id'])

    msg("Copying the oversampled images to the output directory")
    for img in tqdm(oversampled_images):
        shutil.copy(os.path.join(images_dir, img['file_name']), os.path.join(output_dir, img['file_name']))

    oversampled_data = {'info': data['info'], 'licenses': data['licenses'], 'images': oversampled_images,
                        'annotations': oversampled_annotations, 'categories': categories}
    msg("Saving the new oversampled annotations file")
    with open(output_json_path, 'w') as f:
        json.dump(oversampled_data, f)


@wrp
def coco_to_single_class(input_json_path, output_json_path):
    with open(input_json_path) as f:
        data = json.load(f)

    annotations = data['annotations']
    single_class_id = 2

    msg("Modifying category ids")
    for anno in tqdm(annotations):
        anno['category_id'] = single_class_id

    data['annotations'] = annotations
    data['categories'] = [{"id": 1, "name": "Eggs", "supercategory": "none"},
                          {"id": 2, "name": "egg", "supercategory": "Eggs"}]

    msg("Saving the new single class annotations file")
    with open(output_json_path, 'w') as f:
        json.dump(data, f)


@wrp
def fix_coco_annotations(annotation_file):
    with open(annotation_file) as f:
        data = json.load(f)
    cats = [{"id": 1, "name": "Eggs", "supercategory": "none"},
            {"id": 2, "name": "blastula_gastrula", "supercategory": "Eggs"}, {
                "id": 3, "name": "cleavage", "supercategory": "Eggs"},
            {"id": 4, "name": "organogenesis", "supercategory": "Eggs"}]

    if data['categories'] != cats:
        data['categories'] = cats
    else:
        return

    # anno_list = [anno['category_id'] for anno in data['annotations']]
    # mina, maxa = min(anno_list), max(anno_list)
    # mapper = {f"{mina}": 2, f"{mina + 1}": 3, f"{mina + 2}": 4}

    for anno in tqdm(data['annotations']):
        anno['category_id'] += 1

    with open(annotation_file, 'w') as f:
        json.dump(data, f)


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
def analyze_coco_annotations(file_path):
    with open(file_path) as f:
        data = json.load(f)

    num_images = len(data['images'])
    num_annotations = len(data['annotations'])
    num_categories = len(data['categories'])

    image_ids = {img['id'] for img in data['images']}
    annotation_image_ids = {ann['image_id'] for ann in data['annotations']}
    num_images_with_annotations = len(image_ids & annotation_image_ids)

    category_ids = {cat['id']: cat['name'] for cat in data['categories']}
    annotation_category_ids = {ann['category_id']
                               for ann in data['annotations']}
    num_categories_with_annotations = len(
        category_ids.keys() & annotation_category_ids)

    # Count the number of annotations per category
    annotations_per_category = defaultdict(int)
    for ann in tqdm(data['annotations']):
        annotations_per_category[category_ids[ann['category_id']]] += 1

    # Calculate average number of annotations per image
    annotations_per_image = defaultdict(int)
    for ann in tqdm(data['annotations']):
        annotations_per_image[ann['image_id']] += 1
    avg_annotations_per_image = sum(
        annotations_per_image.values()) / num_images

    # Calculate bounding box sizes and aspect ratios
    bounding_box_sizes = []
    bounding_box_aspect_ratios = []
    for ann in tqdm(data['annotations']):
        bbox = ann['bbox']
        size = bbox[2] * bbox[3]  # width * height
        aspect_ratio = bbox[2] / bbox[3]  # width / height
        bounding_box_sizes.append(size)
        bounding_box_aspect_ratios.append(aspect_ratio)

    msg(f'Number of images: {num_images}')
    msg(f'Number of annotations: {num_annotations}')
    msg(f'Number of categories: {num_categories}')
    msg(f'Number of images with annotations: {num_images_with_annotations}')
    msg(
        f'Number of categories with annotations: {num_categories_with_annotations}')
    msg(
        f'Average number of annotations per image: {avg_annotations_per_image}')

    for category, count in annotations_per_category.items():
        msg(f'Number of annotations for category "{category}": {count}')

    msg(
        f'Bounding box sizes: {min(bounding_box_sizes)} / {max(bounding_box_sizes)}')
    msg(
        f'Average bounding box size: {sum(bounding_box_sizes) / len(bounding_box_sizes)}')
    msg(
        f'Bounding box ratios: {min(bounding_box_aspect_ratios)} / {max(bounding_box_aspect_ratios)}')
    msg(
        f'Average bounding box aspect ratio: {sum(bounding_box_aspect_ratios) / len(bounding_box_aspect_ratios)}')


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


@wrp
def xanalyze_annotations(input_json_path, plot=False):
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
    if plot:
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

    pprint(analysis_results)

    return analysis_results


@wrp
def remove_annotation_anomalies(input_json_path):
    # Load the annotations file
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    # Calculate the sizes and aspect ratios of all bounding boxes
    bbox_sizes = []
    aspect_ratios = []
    annotations_per_image = defaultdict(list)
    for anno in data['annotations']:
        bbox = anno['bbox']
        width = bbox[2]
        height = bbox[3]
        bbox_sizes.append(width * height)
        aspect_ratios.append(float(width) / height)
        annotations_per_image[anno['image_id']].append(anno)

    # Convert to numpy arrays for easier calculations
    bbox_sizes = np.array(bbox_sizes)
    aspect_ratios = np.array(aspect_ratios)

    # Calculate the mean and standard deviation of the bounding box sizes and aspect ratios
    size_mean = bbox_sizes.mean()
    size_std = bbox_sizes.std()
    ratio_mean = aspect_ratios.mean()
    ratio_std = aspect_ratios.std()

    # Calculate the mean and standard deviation of the number of annotations per image
    num_annos_per_image = np.array([len(annos) for annos in annotations_per_image.values()])
    anno_count_mean = num_annos_per_image.mean()
    anno_count_std = num_annos_per_image.std()

    # Remove annotations with bounding box sizes or aspect ratios that are more than one standard deviation away from the mean
    # Also remove images with an abnormal number of annotations or where more than 50% of the annotations have been removed
    new_annotations = []
    for image_id, annos in tqdm(annotations_per_image.items()):
        if abs(len(annos) - anno_count_mean) > anno_count_std:
            continue
        new_annos = [anno for anno, size, ratio in zip(annos, bbox_sizes, aspect_ratios)
                     if abs(size - size_mean) <= size_std and abs(ratio - ratio_mean) <= ratio_std]
        if len(new_annos) >= 0.5 * len(annos):
            new_annotations.extend(new_annos)

    data['annotations'] = new_annotations

    # Remove images that no longer have any annotations
    remaining_image_ids = set(anno['image_id'] for anno in data['annotations'])
    data['images'] = [image for image in data['images'] if image['id'] in remaining_image_ids]

    # Save the modified data back to the file
    with open(input_json_path, 'w') as f:
        json.dump(data, f)


@wrp
def clean_annotations_until_threshold(input_json_path, size_std_threshold=10, ratio_std_threshold=0.1,
                                      annotations_per_image_std_threshold=2, max_data_loss=0.2):
    prev_analysis_results = analyze_annotations(input_json_path)
    prev_annotation_count = prev_analysis_results['bbox']['count']

    while True:
        print("Removing anomalies ... ")
        remove_annotation_anomalies(input_json_path)
        analysis_results = analyze_annotations(input_json_path)

        size_std = analysis_results['bbox']['size']['std']
        ratio_std = analysis_results['bbox']['aspect_ratio']['std']
        annotations_per_image_std = analysis_results['annotations_per_image']['std']
        current_annotation_count = analysis_results['bbox']['count']

        if size_std <= size_std_threshold and ratio_std <= ratio_std_threshold and annotations_per_image_std <= annotations_per_image_std_threshold:
            print("Desired thresholds met. Cleaning process completed.")
            break
        elif (prev_annotation_count - current_annotation_count) / prev_annotation_count > max_data_loss:
            print(
                f"Data loss has exceeded the maximum allowed limit of {max_data_loss * 100}%. Cleaning process stopped.")
            break
        else:
            prev_annotation_count = current_annotation_count


@wrp
def suggest_anchors(analysis_results):
    bbox_results = analysis_results['bbox']
    height_results = analysis_results['height']
    width_results = analysis_results['width']
    height_mean = height_results['mean']
    height_std = height_results['std']
    width_mean = width_results['mean']
    width_std = width_results['std']
    aspect_ratio_mean = bbox_results['aspect_ratio']['mean']
    aspect_ratio_std = bbox_results['aspect_ratio']['std']

    # Calculate the geometric mean of the height and width to get the size
    size_mean = np.sqrt(height_mean * width_mean)
    size_std = np.sqrt(height_std * width_std)

    # Suggest 5 anchor sizes around the mean size
    anchor_sizes = [size_mean + (i - 2) * size_std for i in range(5)]
    anchor_sizes = [(int(max(1, size)),) for size in anchor_sizes]  # Ensure sizes are at least 1

    # Suggest 3 anchor ratios around the mean aspect ratio (width / height)
    # aspect_ratio_mean = width_mean / height_mean
    # aspect_ratio_std = width_std / height_std
    anchor_ratios = [aspect_ratio_mean + (i - 1) * aspect_ratio_std for i in range(3)]
    anchor_ratios = [max(0.1, ratio) for ratio in anchor_ratios]  # Ensure ratios are at least 0.1

    msg(f"anchor sizes: {anchor_sizes}")
    msg(f"anchor ratios: {anchor_ratios}")
    return anchor_sizes, anchor_ratios


def run_kmeans_on_sizes(sizes, n_bins=3):
    msg("Running quantile binning on sizes...")
    k_bins = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    labels = k_bins.fit_transform(np.array(sizes).reshape(-1, 1))
    centers = k_bins.bin_edges_[0][1:]  # The first element is -inf, so we skip it
    return labels.flatten(), centers


def get_bounds(values, aggressiveness=1.5):
    msg("Calculating bounds...")
    kde = gaussian_kde(values)
    support = np.linspace(min(values), max(values), 1000)
    density = kde.evaluate(support)
    q1 = np.percentile(density, 25)
    q3 = np.percentile(density, 75)
    iqr = q3 - q1
    lower_bound = q1 - (iqr * aggressiveness)
    upper_bound = q3 + (iqr * aggressiveness)
    msg(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    return lower_bound, upper_bound


def transform_sizes(sizes):
    return np.log(sizes)


def quantile_binning_on_sizes(sizes, n_bins=3):
    msg("Running quantile binning on sizes...")
    k_bins = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    labels = k_bins.fit_transform(np.array(sizes).reshape(-1, 1))
    centers = k_bins.bin_edges_[0][1:]  # The first element is -inf, so we skip it
    return labels.flatten(), centers


def remove_and_resize_outliers(annotations, aspect_ratio_bounds, size_bounds, cluster_centers):
    msg("Removing and resizing outliers...")

    new_aspect_ratios = []
    new_sizes = []
    new_annotations = []
    resized_boxes = defaultdict(list)
    outliers = defaultdict(list)

    for idx, annotation in tqdm(enumerate(annotations['annotations']), total=len(annotations['annotations'])):
        bbox = annotation['bbox']
        size = bbox[2] * bbox[3]  # Change here to consider the area of the box
        aspect_ratio = bbox[2] / bbox[3]

        msg(f"Original bbox: {bbox}")

        # Check if the original aspect ratio is within the bounds
        if aspect_ratio < aspect_ratio_bounds[0] or aspect_ratio > aspect_ratio_bounds[1]:
            outliers[annotation['image_id']].append(annotation)
            continue

        if size < size_bounds[0] or size > size_bounds[1]:
            outliers[annotation['image_id']].append(annotation)
        else:
            closest_center = min(cluster_centers, key=lambda x: abs(x - size))
            # Resize only the dimensions of the box, keep the central point
            center = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
            new_width = math.sqrt(closest_center * aspect_ratio)  # Adjusted to maintain aspect ratio
            new_height = math.sqrt(closest_center / aspect_ratio)  # Adjusted to maintain aspect ratio
            resized_bbox = [center[0] - new_width / 2, center[1] - new_height / 2, new_width, new_height]

            msg(f"Resized box: {resized_bbox}")

            resized_boxes[annotation['image_id']].append(resized_bbox)

            new_aspect_ratios.append(aspect_ratio)
            new_sizes.append(closest_center)
            new_annotations.append(annotation)

    return new_aspect_ratios, new_sizes, new_annotations, resized_boxes, outliers


def remove_outliers(annotations_file, image_dir, output_dir, aggressiveness=1.0, ratio_bounds=None, size_bounds=None,
                    rm_overlap=True, size_occurances=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    image_lookup = {image['id']: image['file_name'] for image in annotations['images']}

    aspect_ratios = []
    sizes = []
    for annotation in annotations['annotations']:
        bbox = annotation['bbox']
        aspect_ratio = bbox[2] / bbox[3]
        size = bbox[2] * bbox[3]  # Change here to consider the area of the box
        aspect_ratios.append(aspect_ratio)
        sizes.append(size)

    if ratio_bounds is None:
        ratio_bounds = get_bounds(aspect_ratios, aggressiveness)
    if size_bounds is None:
        sizes = transform_sizes(sizes)
        size_bounds = get_bounds(sizes, aggressiveness)

    cluster_labels, cluster_centers = quantile_binning_on_sizes(sizes)

    annotations = balance_box_areas(annotations, sizes)

    new_aspect_ratios, new_sizes, new_annotations, resized_boxes, outliers = remove_and_resize_outliers(annotations,
                                                                                                        ratio_bounds,
                                                                                                        size_bounds,
                                                                                                        cluster_labels,
                                                                                                        cluster_centers)

    for image_id, boxes in resized_boxes.items():
        for box in boxes:
            new_annotations.append({
                'image_id': image_id,
                'bbox': box
            })

    modified_images = Parallel(n_jobs=-1)(
        delayed(draw_over_outliers)(image_id, outliers, image_lookup, image_dir) for image_id in
        tqdm(outliers.keys(), total=len(outliers.keys())))

    for image_id, image in zip(outliers.keys(), modified_images):
        output_file = os.path.join(output_dir, image_lookup[image_id])
        cv2.imwrite(output_file, image)

    output_annotations_file = os.path.join(output_dir, 'annotations.json')
    with open(output_annotations_file, 'w') as f:
        json.dump(annotations, f)


def balance_box_areas(annotations, sizes, n_bins=5):
    msg("Balancing box areas...")

    # First we bin the sizes into n_bins categories
    k_bins = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    size_bins = k_bins.fit_transform(np.array(sizes).reshape(-1, 1)).flatten().astype(int)

    # Count number of boxes in each bin
    size_bin_counts = Counter(size_bins)

    # Get the max count across all bins to be our target
    target_count = max(size_bin_counts.values())

    max_id = max(annotation['id'] for annotation in annotations['annotations'])

    # Now we compute the deficit for each bin
    for bin_id in range(n_bins):
        bin_count = size_bin_counts[bin_id]
        deficit = target_count - bin_count
        if deficit > 0:
            boxes_in_bin = np.where(size_bins == bin_id)[0]
            if len(boxes_in_bin) == 0:
                msg(f"No boxes in bin {bin_id}. Skipping...")
                continue

            np.random.shuffle(boxes_in_bin)
            boxes_to_duplicate = np.random.choice(boxes_in_bin, size=deficit)

            for box_id in boxes_to_duplicate:
                max_id += 1
                duplicate = dict(annotations['annotations'][box_id])
                duplicate['id'] = max_id
                annotations['annotations'].append(duplicate)
    return annotations


def draw_over_outliers(image_id, outliers, image_lookup, image_dir, avg_clr=False):
    if image_id not in outliers:
        return None

    image_file = os.path.join(image_dir, image_lookup[image_id])
    image = cv2.imread(image_file)

    for outlier in outliers[image_id]:
        x, y, w, h = [int(coord) for coord in outlier['bbox']]
        height, width, _ = image.shape
        x = min(x, width - 1)
        y = min(y, height - 1)
        w = min(w, width - x)
        h = min(h, height - y)

        # Get the border pixels of the bounding box
        top_border = image[y, x:x + w]
        bottom_border = image[min(y + h, height - 1), x:x + w]
        left_border = image[y:y + h, x]
        right_border = image[y:y + h, min(x + w, width - 1)]

        # Calculate the average color of the border pixels
        border_pixels = np.concatenate((top_border, bottom_border, left_border, right_border), axis=0)
        avg_color = np.mean(border_pixels, axis=0)

        if not avg_clr:
            avg_color = (0, 0, 0)

        cv2.rectangle(image, (x, y), (x + w, y + h), avg_color, -1)

    return image


def display_random_image_with_boxes(json_file, times=3):
    # Define a list of predefined colors for different classes
    class_colors = ['c', 'r', 'g', 'b', 'm', 'y']

    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    for _ in range(times):
        # Select a random image from the dataset
        image_info = random.choice(data['images'])
        image_path = os.path.join(os.path.dirname(json_file), image_info['file_name'])

        # Find the corresponding annotations for the selected image
        annotations = [anno for anno in data['annotations'] if anno['image_id'] == image_info['id']]

        # Load the image using OpenCV
        image = cv2.imread(image_path)

        # Check if the image is not empty
        if image is not None:
            # Create figure and axes
            fig, ax = plt.subplots(1)

            # Display the image
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Draw bounding boxes on the image
            for annotation in annotations:
                bbox = annotation['bbox']
                x, y, w, h = bbox
                class_id = annotation['category_id']

                # Select color for the class
                color = class_colors[class_id % len(class_colors)]

                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

            # Show the image with bounding boxes
            plt.show()
        else:
            msg("Failed to load the image:", image_path)


@wrp
def check_image_sizes(annotation_file, images_dir):
    # Load annotation data
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    # Create a dictionary of image resolutions and their count
    image_sizes = {}
    image_file_sizes = {}
    for img in tqdm(data['images'], desc="Checking image sizes"):
        image_file = os.path.join(images_dir, img['file_name'])
        with Image.open(image_file) as image:
            # Get the image size
            size = image.size
            # Increment the count for this size
            image_sizes[size] = image_sizes.get(size, 0) + 1
            # Store the image file size
            image_file_sizes[image_file] = size

    # Find the most common image size
    most_common_size = max(image_sizes, key=image_sizes.get)

    if len(image_sizes) != 1:
        msg("Removing images that are not of the most common size")
        new_images = [img for img in data['images'] if
                      image_file_sizes[os.path.join(images_dir, img['file_name'])] == most_common_size]

        # Update the 'images' field in the annotation data
        data['images'] = new_images

        # Update the 'annotations' field in the annotation data
        # to include only those annotations that correspond to the remaining images
        remaining_image_ids = {img['id'] for img in new_images}
        data['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] in remaining_image_ids]

        # Save the modified annotation data
        with open(annotation_file, 'w') as f:
            json.dump(data, f)

        # Delete images that are not of the most common size
        for image_file, size in image_file_sizes.items():
            if size != most_common_size:
                os.remove(image_file)


def freedman_diaconis(data):
    """Determine number of bins using Freedman-Diaconis rule."""
    data_range = np.ptp(data)
    bin_width = 2 * iqr(data) / np.power(len(data), 1 / 3)
    return int(np.ceil(data_range / bin_width))


@wrp
def fix_size_outliers(annotation_file):
    # Load annotation data
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    sizes = []
    aspect_ratios = []
    for annotation in data['annotations']:
        bbox = annotation['bbox']
        width = bbox[2]
        height = bbox[3]
        sizes.append(width * height)
        aspect_ratios.append(width / height)

    # Convert to numpy arrays for easier manipulation
    sizes = np.array(sizes)
    aspect_ratios = np.array(aspect_ratios)

    # Calculate number of bins dynamically using Freedman-Diaconis rule
    n_bins = freedman_diaconis(sizes)

    # Bin indices for each element in sizes
    bin_indices = np.digitize(sizes, bins=np.histogram(sizes, bins=n_bins)[1])

    # Adjust sizes
    for bin_i in tqdm(range(1, n_bins + 1), desc='Adjusting sizes'):
        indices_in_bin = np.where(bin_indices == bin_i)[0]
        if len(indices_in_bin) > 0:
            median_size_in_bin = np.median(sizes[indices_in_bin])

            for idx in indices_in_bin:
                original_bbox = data['annotations'][idx]['bbox']
                original_size = sizes[idx]
                original_aspect_ratio = aspect_ratios[idx]

                # Compute new width and height while maintaining the aspect ratio
                new_width = math.sqrt(median_size_in_bin * original_aspect_ratio)
                new_height = math.sqrt(median_size_in_bin / original_aspect_ratio)

                # Compute new top left corner to maintain the center position
                center_x = original_bbox[0] + original_bbox[2] / 2
                center_y = original_bbox[1] + original_bbox[3] / 2
                new_x = center_x - new_width / 2
                new_y = center_y - new_height / 2

                # Replace the original bbox in the annotation
                data['annotations'][idx]['bbox'] = [new_x, new_y, new_width, new_height]

    # Save the modified annotation data
    with open(annotation_file, 'w') as f:
        json.dump(data, f)


@wrp
def simple_process(root_dit, version=7, num_images=0, num_augmented_images=10):
    json_file = r"\annotations.json"
    orig_dir = root_dir + r"\original"
    orig_json = orig_dir + json_file
    ovr_dir = root_dir + rf"\ovr_{version}"
    ovr_json = ovr_dir + json_file
    div_dir = root_dir + rf"\div_{version}"
    div_json = div_dir + json_file
    blc_dir = root_dir + rf"\blc_{version}"
    blc_json = blc_dir + json_file
    blc_sjson = blc_dir + r"\sc_annotations.json"
    aug_dir = root_dir + rf"\aug_{version}"
    aug_json = aug_dir + json_file

    augment_dataset(orig_dir, orig_json, aug_dir, num_augmented_images, simple=True)
    undersample_coco(aug_json, aug_dir, blc_dir, blc_json)
    coco_to_single_class(blc_json, blc_sjson)


@wrp
def xremove_outliers(annotations_file, image_dir, output_dir, aggressiveness=1.0, ratio_bounds=None, rm_overlap=True):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    msg(
        f"Running remove_outliers with:\nannotations_file: {annotations_file}\nimage_dir: {image_dir}\noutput_dir: {output_dir}\naggressiveness: {aggressiveness}\nratio_bounds: {ratio_bounds}")

    # Load the annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    # Create a lookup dictionary for image filenames
    image_lookup = {image['id']: image['file_name'] for image in annotations['images']}

    # Calculate aspect ratios and sizes for all bounding boxes
    aspect_ratios = []
    sizes = []
    for annotation in tqdm(annotations['annotations']):
        bbox = annotation['bbox']
        try:
            aspect_ratio = bbox[2] / bbox[3]
        except ZeroDivisionError:
            aspect_ratio = 0
        size = bbox[2] * bbox[3]
        aspect_ratios.append(aspect_ratio)
        sizes.append(size)

    # Define a function to calculate the upper and lower bounds for outliers
    def get_bounds(values):

        def zscore(values):
            mean = np.mean(values)
            std = np.std(values)
            lower_bound = mean - (1 / aggressiveness) * std
            upper_bound = mean + (1 / aggressiveness) * std
            return lower_bound, upper_bound

        def mad(values):
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            lower_bound = median - (1 / aggressiveness) * mad
            upper_bound = median + (1 / aggressiveness) * mad
            return lower_bound, upper_bound

        if aggressiveness:
            zl, zu = zscore(values)  # high
            ml, mu = mad(values)  # low
            msg(f"Zscore: {zl, zu}\nMAD score: {ml, mu}")
            return ml, zu

        msg(f"Skipping size removals ...")
        return 0, 999999999

    # Get the bounds for sizes
    size_bounds = get_bounds(sizes)
    msg(f"Size bounds: {size_bounds}")

    # Create a dictionary to keep track of modified images and their outliers
    outliers = defaultdict(list)
    modified_images = {}

    # Initialize the standard deviation for aspect ratios
    std_dev = np.std(aspect_ratios)

    while True:
        # Get the bounds for aspect ratios
        aspect_ratio_bounds = get_bounds(aspect_ratios) if ratio_bounds is None else ratio_bounds

        # Initialize the lists for the new aspect ratios, sizes, and annotations
        new_aspect_ratios = []
        new_sizes = []
        new_annotations = []
        new_images = []

        # Iterate over the annotations
        for annotation in tqdm(annotations['annotations']):
            # If the aspect ratio or size is an outlier, remove the annotation and add it to the outliers
            try:
                aspect_ratio = annotation['bbox'][2] / annotation['bbox'][3]
            except ZeroDivisionError:
                aspect_ratio = 0
            size = annotation['bbox'][2] * annotation['bbox'][3]
            if (aspect_ratio < aspect_ratio_bounds[0] or aspect_ratio > aspect_ratio_bounds[1] or
                    size < size_bounds[0] or size > size_bounds[1]):
                # Add the annotation to the outliers
                outliers[annotation['image_id']].append(annotation)
            else:
                # If the aspect ratio and size are not outliers, add them to the new lists
                new_aspect_ratios.append(aspect_ratio)
                new_sizes.append(size)
                new_annotations.append(annotation)

        # Replace the old annotations with the new ones
        annotations['annotations'] = new_annotations

        # Remove images with no annotations
        image_ids = set([anno['image_id'] for anno in new_annotations])
        new_images = [img for img in annotations['images'] if img['id'] in image_ids]
        annotations['images'] = new_images

        # Check if the standard deviation of the aspect ratios has decreased significantly
        new_std_dev = np.std(new_aspect_ratios)
        if new_std_dev >= std_dev * 0.95:  # Stop if the standard deviation has not decreased by at least 5%
            break

        # Update the aspect ratios, sizes, and standard deviation
        aspect_ratios = new_aspect_ratios
        sizes = new_sizes
        std_dev = new_std_dev

    # # Draw over the outliers in the modified images
    # for image_id, image_outliers in outliers.items():
    #     # Load the image if it hasn't been loaded yet
    #     if image_id not in modified_images:
    #         image_file = os.path.join(image_dir, image_lookup[image_id])
    #         modified_images[image_id] = cv2.imread(image_file)
    #
    #     # Draw over each outlier in the image
    #     for outlier in image_outliers:
    #         # Get the bounding box coordinates
    #         x, y, w, h = [int(coord) for coord in outlier['bbox']]
    #
    #         # Ensure the indices are within the image dimensions
    #         height, width, _ = modified_images[image_id].shape
    #         x = min(x, width - 1)
    #         y = min(y, height - 1)
    #         w = min(w, width - x)
    #         h = min(h, height - y)
    #
    #         # Get the border pixels of the bounding box
    #         top_border = modified_images[image_id][y, x:x + w]
    #         bottom_border = modified_images[image_id][min(y + h, height - 1), x:x + w]
    #         left_border = modified_images[image_id][y:y + h, x]
    #         right_border = modified_images[image_id][y:y + h, min(x + w, width - 1)]
    #
    #         # Calculate the average color of the border pixels
    #         border_pixels = np.concatenate((top_border, bottom_border, left_border, right_border), axis=0)
    #         avg_color = np.mean(border_pixels, axis=0)
    #
    #         # Draw over the bounding box with the average color
    #         cv2.rectangle(modified_images[image_id], (x, y), (x + w, y + h), avg_color, -1)
    #
    # # Save all modified images
    # for image_id, image in modified_images.items():
    #     output_file = os.path.join(output_dir, image_lookup[image_id])
    #     cv2.imwrite(output_file, image)

    # Save the modified annotations in the new images directory
    output_annotations_file = os.path.join(output_dir, 'annotations.json')
    with open(output_annotations_file, 'w') as f:
        json.dump(annotations, f)

@wrp
def remove_overlapping_annotations(file_path: str, threshold: float = 0.5) -> None:
    """Remove overlapping annotations from a JSON COCO format file."""

    msg("Removing overlapping annotations ...")
    with open(file_path, 'r') as f:
        data = json.load(f)

    def calculate_iou(box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) for two bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area if union_area != 0 else 0

        return iou

    idx = index.Index()
    for i, annotation in enumerate(data['annotations']):
        bbox = annotation['bbox']
        # Convert the bounding box coordinates to the format expected by the R-tree index
        idx.insert(i, (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

    new_annotations = []
    for i, annotation in (enumerate(q(data['annotations']))):
        bbox = annotation['bbox']
        overlap = False
        # Convert the bounding box coordinates to the format expected by the R-tree index
        bbox = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
        for j in idx.intersection(bbox):
            if i != j and calculate_iou(bbox, data['annotations'][j]['bbox']) > threshold:
                overlap = True
                break
        if not overlap:
            new_annotations.append(annotation)

    data['annotations'] = new_annotations

    with open(file_path, 'w') as f:
        json.dump(data, f)
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


@wrp
def check_image_sizes(annotation_file, images_dir):
    # Load annotation data
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    # Create a dictionary of image resolutions and their count
    image_sizes = {}
    for img in data['images']:
        image_file = os.path.join(images_dir, img['file_name'])
        with Image.open(image_file) as image:
            # Get the image size
            size = image.size
            # Increment the count for this size
            image_sizes[size] = image_sizes.get(size, 0) + 1

    # Find the most common image size
    most_common_size = max(image_sizes, key=image_sizes.get)
    print(f"Image sizes: {set(s for s in image_sizes.keys())}")
    print("Most common image size: ", most_common_size)
    return
    if len(image_sizes) == 1:
        msg("Remove images that are not of the most common size")
        new_images = []
        for img in tqdm(data['images']):
            image_file = os.path.join(images_dir, img['file_name'])
            with Image.open(image_file) as image:
                if image.size != most_common_size:
                    # Remove the image file
                    os.remove(image_file)
                else:
                    new_images.append(img)

        # Update the 'images' field in the annotation data
        data['images'] = new_images

        # Update the 'annotations' field in the annotation data
        # to include only those annotations that correspond to the remaining images
        remaining_image_ids = {img['id'] for img in new_images}
        data['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] in remaining_image_ids]

        # Save the modified annotation data
        with open(annotation_file, 'w') as f:
            json.dump(data, f)


def process_sizes(size, max_count, size_dict):
    # Upscale each size in a separate process
    if len(size_dict[size]) < max_count:
        duplicates = random.choices(size_dict[size], k=max_count - len(size_dict[size]))
        return duplicates
    else:
        return []


@wrp
def xupsample_sizes(annotation_file, n_jobs=-1):
    # Load annotation data
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    # Calculate bounding box sizes and count occurrences
    boxes = np.array([ann['bbox'] for ann in data['annotations']])
    sizes = np.round(boxes[:, 2] * boxes[:, 3], 2)
    size_counts = Counter(sizes)

    # Find the maximum count
    max_count = max(size_counts.values())

    # Create a dictionary of annotations grouped by their bounding box size
    size_dict = defaultdict(list)
    for ann in tqdm(data['annotations'], desc='annotations grouping...'):
        size = np.round(ann['bbox'][2] * ann['bbox'][3], 2)
        size_dict[size].append(ann)

    msg("Upsampling bounding boxes...")
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(delayed(process_sizes)(size, max_count, size_dict) for size in tqdm(size_counts.keys()))

    # Flatten the list of lists
    duplicates = [item for sublist in results for item in sublist]

    # Add duplicates to annotations
    data['annotations'].extend(duplicates)

    # Save the updated annotation data
    with open(annotation_file, 'w') as f:
        json.dump(data, f)
    msg(f"Upsampling completed. The annotations have been updated in {annotation_file}")


@wrp
def remove_abnormal_images(annotation_file, image_dir, std_multiplier=2):
    # Load annotation data
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    # Count the number of bounding boxes for each image
    image_to_box_count = defaultdict(int)
    for ann in data['annotations']:
        image_to_box_count[ann['image_id']] += 1

    # Calculate the mean and standard deviation
    counts = list(image_to_box_count.values())
    mean_count = mean(counts)
    stdev_count = stdev(counts)

    # Define the threshold
    threshold = mean_count + std_multiplier * stdev_count

    # Find image IDs to be removed
    ids_to_remove = [id for id, count in image_to_box_count.items() if count > threshold]

    # Delete actual images
    for img in q(data['images']):
        if img['id'] in ids_to_remove:
            image_path = os.path.join(image_dir, img['file_name'])
            if os.path.exists(image_path):
                os.remove(image_path)

    # Remove images from annotations
    data['images'] = [img for img in data['images'] if img['id'] not in ids_to_remove]

    # Remove corresponding annotations
    data['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] not in ids_to_remove]

    # Save the modified annotation data
    with open(annotation_file, 'w') as f:
        json.dump(data, f)

    msg(f'Removed {len(ids_to_remove)} images from annotations.')


def draw_over_annotations(img, ann):
    # # Get bounding box coordinates
    # x, y, w, h = [int(coord) for coord in ann['bbox']]
    # # Ensure the indices are within the image dimensions
    # height, width, _ = img.shape
    # x = min(x, width - 1)
    # y = min(y, height - 1)
    # w = min(w, width - x)
    # h = min(h, height - y)
    # # Get the border pixels of the bounding box
    # top_border = img[y, x:x + w]
    # bottom_border = img[min(y + h, height - 1), x:x + w]
    # left_border = img[y:y + h, x]
    # right_border = img[y:y + h, min(x + w, width - 1)]
    # # Calculate the average color of the border pixels
    # border_pixels = np.concatenate((top_border, bottom_border, left_border, right_border), axis=0)
    # avg_color = np.mean(border_pixels, axis=0)
    # # Draw over the bounding box with the average color
    # cv2.rectangle(img, (x, y), (x + w, y + h), tuple(avg_color), -1)

    return img


@wrp
def xupsample_sizes(annotation_file, images_dir, max_count, image_dup_thresh=5, n_jobs=-1):
    # Load annotation data
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    for ann in data['annotations']:
        assert any(img for img in data['images'] if img['id'] == ann['image_id']), "No image found for annotation"

    # Calculate bounding box sizes and count occurrences
    boxes = np.array([ann['bbox'] for ann in data['annotations']])
    sizes = np.round(boxes[:, 2] * boxes[:, 3], 2)
    size_counts = Counter(sizes)

    # Create a dictionary of annotations and images grouped by their bounding box size
    size_dict = defaultdict(list)
    for ann in data['annotations']:
        size = np.round(ann['bbox'][2] * ann['bbox'][3], 2)
        size_dict[size].append(ann)

    msg("Stage 1: Duplicate entire images along with relevant annotations")
    # Create a copy of size_dict to keep track of sizes still needing more occurrences
    size_dict_copy = size_dict.copy()
    duplicates = []
    max_img_id = max(img['id'] for img in data['images'])

    for size, annotations in q(size_dict_copy.items(), desc='Copying ...'):
        if len(annotations) < max_count:
            for _ in range(min(max_count - len(annotations), image_dup_thresh)):
                ann = random.choice(annotations)
                original_img = next((img for img in data['images'] if img['id'] == ann['image_id']), None)
                if original_img is None:
                    print(f"No matching image found for image_id {ann['image_id']}")
                    continue

                new_img_id = max_img_id + 1
                max_img_id += 1
                base_name, ext = os.path.splitext(original_img['file_name'])
                new_file_name = f"{base_name}_dup_{new_img_id}{ext}"
                new_file_path = os.path.join(images_dir, new_file_name)

                # Copy the original image to the new image file
                shutil.copy(os.path.join(images_dir, original_img['file_name']), new_file_path)

                # Load the new image and draw over irrelevant annotations
                new_img = cv2.imread(new_file_path)
                for irrelevant_ann in [a for a in data['annotations'] if
                                       a['image_id'] == ann['image_id'] and a['id'] != ann['id']]:
                    new_img = draw_over_annotations(new_img, irrelevant_ann)
                cv2.imwrite(new_file_path, new_img)

                # Create new annotation and image records
                new_ann = copy.deepcopy(ann)
                new_ann['image_id'] = new_img_id
                new_ann['id'] = len(data['annotations']) + len(duplicates) + 1
                duplicates.append(new_ann)

                new_image = copy.deepcopy(original_img)
                new_image['id'] = new_img_id
                new_image['file_name'] = new_file_name
                data['images'].append(new_image)  # Add new image to data['images'] immediately
                size_dict_copy[size].append(new_ann)

    msg("Stage 2: Duplicate annotations in the same images")
    for size, annotations in q(size_dict_copy.items(), desc='Duplicating annotations ...'):
        if len(annotations) < max_count:
            for _ in range(max_count - len(annotations)):
                ann = random.choice(annotations)
                new_ann = copy.deepcopy(ann)
                new_ann['id'] = len(data['annotations']) + len(duplicates) + 1
                duplicates.append(new_ann)

    # Add new duplicates to annotations
    data['annotations'].extend(duplicates)

    # Save the modified annotation data
    with open(annotation_file, 'w') as f:
        json.dump(data, f)

    msg(f'Upsampling done. The maximum count of each size is now {max_count}.')


@wrp
def xxupsample_sizes(annotation_file, images_dir, max_count, image_dup_thresh=5, count_for_dup=10):
    # Load annotation data
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    assert all(any(img for img in data['images'] if img['id'] == ann['image_id']) for ann in
               data['annotations']), "No image found for annotation"

    # Calculate bounding box sizes and count occurrences
    sizes = np.array([ann['bbox'][2] * ann['bbox'][3] for ann in data['annotations']])
    sizes = np.round(sizes, 2)

    # Create a dictionary of annotations and images grouped by their bounding box size
    size_dict = defaultdict(list)
    for ann in data['annotations']:
        size = np.round(ann['bbox'][2] * ann['bbox'][3], 2)
        size_dict[size].append(ann)

    duplicates = []
    max_img_id = max(img['id'] for img in data['images'])

    if (max_count >= count_for_dup) and False:
        print("Stage 1: Duplicate entire images along with relevant annotations")
        # Create a copy of size_dict to keep track of sizes still needing more occurrences
        size_dict_copy = size_dict.copy()

        for size, annotations in q(size_dict_copy.items(), desc='Copying ...'):
            if len(annotations) < max_count:
                for idx in range(min(max_count - len(annotations), image_dup_thresh)):
                    ann = annotations[idx % len(annotations)]
                    original_img = next((img for img in data['images'] if img['id'] == ann['image_id']), None)

                    if original_img is None:
                        print(f"No matching image found for image_id {ann['image_id']}")
                        continue

                    new_img_id = max_img_id + 1
                    max_img_id += 1
                    base_name, ext = os.path.splitext(original_img['file_name'])
                    new_file_name = f"{base_name}_dup_{new_img_id}{ext}"
                    new_file_path = os.path.join(images_dir, new_file_name)

                    # Load the original image and draw over irrelevant annotations
                    img_path = os.path.join(images_dir, original_img['file_name'])
                    img = cv2.imread(img_path)
                    for irrelevant_ann in [a for a in data['annotations'] if
                                           a['image_id'] == ann['image_id'] and a['id'] != ann['id']]:
                        img = draw_over_annotations(img, irrelevant_ann)
                    cv2.imwrite(new_file_path, img)

                    # Create new annotation and image records
                    new_ann = copy.deepcopy(ann)
                    new_ann['image_id'] = new_img_id
                    new_ann['id'] = len(data['annotations']) + len(duplicates) + 1
                    duplicates.append(new_ann)

                    new_image = copy.deepcopy(original_img)
                    new_image['id'] = new_img_id
                    new_image['file_name'] = new_file_name
                    data['images'].append(new_image)
                    size_dict_copy[size].append(new_ann)

    print("Stage 2: Duplicate annotations in the same images")
    size_dict_copy = size_dict.copy()

    for size, annotations in q(size_dict_copy.items(), desc='Duplicating annotations ...'):
        if len(annotations) < max_count:
            for idx in range(max_count - len(annotations)):
                ann = annotations[idx % len(annotations)]
                new_ann = copy.deepcopy(ann)
                new_ann['id'] = len(data['annotations']) + len(duplicates) + 1
                duplicates.append(new_ann)

    # Add new duplicates to annotations
    data['annotations'].extend(duplicates)

    # Save the modified annotation data
    with open(annotation_file, 'w') as f:
        json.dump(data, f)

    print(f'Upsampling done. The maximum count of each size is now {max_count}.')


def duplicate_image_flip(img, ann, images_dir, max_img_id, data):
    # Generate a random number to decide the flip/rotate operation
    flip_op = np.random.randint(3)

    if flip_op == 0:  # Flip horizontally
        seq = iaa.Sequential([iaa.Fliplr(1)])  # Always horizontally flip each input image
        new_img = seq(image=img)
        new_ann = ann.copy()
        new_ann['bbox'][0] = img.shape[1] - ann['bbox'][0] - ann['bbox'][2]  # Adjust x coordinate
    elif flip_op == 1:  # Flip vertically
        seq = iaa.Sequential([iaa.Flipud(1)])  # Always vertically flip each input image
        new_img = seq(image=img)
        new_ann = ann.copy()
        new_ann['bbox'][1] = img.shape[0] - ann['bbox'][1] - ann['bbox'][3]  # Adjust y coordinate
    else:  # Rotate 180 degrees
        seq = iaa.Sequential([iaa.Rotate(180)])
        new_img = seq(image=img)
        new_ann = ann.copy()  # For rotation, the bounding box remains the same

    # Create a new image ID and file name
    new_img_id = max_img_id + 1
    original_img = next((img for img in data['images'] if img['id'] == ann['image_id']), None)
    base_name, ext = os.path.splitext(original_img['file_name'])
    new_file_name = f"{base_name}_flip_{new_img_id}{ext}"
    new_file_path = os.path.join(images_dir, new_file_name)

    # Save the new image to disk
    cv2.imwrite(new_file_path, new_img)

    # Update the annotation with the new image ID and adjusted bounding box
    new_ann['image_id'] = new_img_id
    new_ann['id'] = new_img_id  # We can simply use the new image ID as the annotation ID

    return new_ann, new_img_id, new_file_name


def calc_farness(current_count, target_count, threshold=0.5):
    return current_count / target_count < threshold


@wrp
def upsample_sizes(annotation_file, images_dir, max_count, max_inplace_dup=3, threshold=0.5):
    # Load annotation data
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    assert all(any(img for img in data['images'] if img['id'] == ann['image_id']) for ann in
               data['annotations']), "No image found for annotation"

    # Calculate bounding box sizes and count occurrences
    sizes = np.array([ann['bbox'][2] * ann['bbox'][3] for ann in data['annotations']])
    sizes = np.round(sizes, 2)

    # Create a dictionary of annotations and images grouped by their bounding box size
    size_dict = defaultdict(list)
    for ann in data['annotations']:
        size = np.round(ann['bbox'][2] * ann['bbox'][3], 2)
        size_dict[size].append(ann)

    duplicates = []
    max_img_id = max(img['id'] for img in data['images'])

    for size, annotations in q(size_dict.items(), desc='Duplicating ...'):
        # Calculate farness
        is_far = calc_farness(len(annotations), max_count, threshold)

        # Duplicate annotations in place if far from the target
        if is_far:
            inplace_dup_count = min(max_count - len(annotations), max_inplace_dup)
            for idx in range(inplace_dup_count):
                ann = annotations[idx % len(annotations)]
                new_ann = copy.deepcopy(ann)
                new_ann['id'] = len(data['annotations']) + len(duplicates) + 1
                duplicates.append(new_ann)

        # Duplicate images if near to the target or if inplace duplications have been maxed out
        img_dup_count = max_count - len(annotations) - len(duplicates)
        for idx in range(img_dup_count):
            ann = annotations[idx % len(annotations)]
            original_img = next((img for img in data['images'] if img['id'] == ann['image_id']), None)

            if original_img is None:
                print(f"No matching image found for image_id {ann['image_id']}")
                continue

            # Load the original image and draw over irrelevant annotations
            img_path = os.path.join(images_dir, original_img['file_name'])
            img = cv2.imread(img_path)
            for irrelevant_ann in [a for a in data['annotations'] if
                                   a['image_id'] == ann['image_id'] and a['id'] != ann['id']]:
                img = draw_over_annotations(img, irrelevant_ann)

            new_ann, max_img_id, new_file_name = duplicate_image_flip(img, ann, images_dir, max_img_id, data)

            # Create new annotation and image records
            new_ann['id'] = len(data['annotations']) + len(duplicates) + 1
            duplicates.append(new_ann)

            new_image = copy.deepcopy(original_img)
            new_image['id'] = new_ann['image_id']
            new_image['file_name'] = new_file_name
            data['images'].append(new_image)

    # Add new duplicates to annotations
    data['annotations'].extend(duplicates)

    # Save the modified annotation data
    with open(annotation_file, 'w') as f:
        json.dump(data, f)

    print(f'Upsampling done. The maximum count of each size is now {max_count}.')


@wrp
def fdownsample_sizes(annotation_file, images_dir, min_count):
    # Load annotation data
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    # Calculate bounding box sizes and count occurrences
    size_dict = defaultdict(list)
    for ann in data['annotations']:
        box = ann['bbox']
        size = box[2] * box[3]
        size_dict[size].append(ann['id'])

    # Record annotations to remove and their corresponding images
    to_remove = []
    modified_images = {}

    for size, ids in q(size_dict.items(), desc='Records to remove'):
        if len(ids) > min_count:
            excess = random.sample(ids, len(ids) - min_count)
            to_remove.extend(excess)
            for ann_id in excess:
                for ann in data['annotations']:
                    if ann['id'] == ann_id:
                        img = modified_images.get(ann['image_id'])
                        if img is None or img.size == 0:
                            image_file = os.path.join(images_dir, next(
                                img['file_name'] for img in data['images'] if img['id'] == ann['image_id']))
                            img = cv2.imread(image_file)
                            modified_images[ann['image_id']] = img

                        # Draw over the bounding box with the average color
                        modified_images[ann['image_id']] = draw_over_annotations(img, ann)

    # Save the modified images
    for img_id, img in q(modified_images.items(), desc='Save the modified images'):
        img_file = os.path.join(images_dir,
                                next((image['file_name'] for image in data['images'] if image['id'] == img_id), None))
        cv2.imwrite(img_file, img)

    # Remove excess annotations
    data['annotations'] = [ann for ann in data['annotations'] if ann['id'] not in to_remove]

    # Remove images without any annotations
    data['images'] = [img for img in data['images'] if
                      any(ann for ann in data['annotations'] if ann['image_id'] == img['id'])]

    # Save the modified annotation data
    with open(annotation_file, 'w') as f:
        json.dump(data, f)

    msg(f'Downsampling done. The minimum count of each size is now {min_count}.')


@wrp
def xxdownsample_sizes(annotation_file, min_count):
    # Load annotation data
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    # Calculate bounding box sizes and count occurrences
    size_dict = defaultdict(list)
    for ann in data['annotations']:
        box = ann['bbox']
        size = round(box[2] * box[3], 2)
        size_dict[size].append(ann)

    # Record annotations to remove
    to_remove = []

    for size, anns in q(size_dict.items(), desc='Removing annotations...'):
        if len(anns) > min_count:
            excess_anns = random.sample(anns, len(anns) - min_count)
            to_remove.extend([ann['id'] for ann in excess_anns])

    # Filter out the annotations to remove
    data['annotations'] = [ann for ann in data['annotations'] if ann['id'] not in to_remove]

    # Filter out the images that have no annotations
    data['images'] = [img for img in data['images'] if any(ann['image_id'] == img['id'] for ann in data['annotations'])]

    # Save the modified annotation data
    with open(annotation_file, 'w') as f:
        json.dump(data, f)

    print(f'Downsampling done. The minimum count of each size is now {min_count}.')


def get_random_image_dimensions(directory):
    # Get the list of image files in the directory
    image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Select a random image file
    random_image_file = image_files[0]

    # Read the image using OpenCV
    image_path = os.path.join(directory, random_image_file)
    image = cv2.imread(image_path)

    # Get the dimensions of the image
    height, width, _ = image.shape

    return width, height


@wrp
def downsample_sizes(annotation_file, images_dir, min_count, noise_stddev=0.01):
    # Load annotation data
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    width, height = get_random_image_dimensions(images_dir)
    # Calculate bounding box sizes and count occurrences
    size_dict = defaultdict(list)
    for ann in data['annotations']:
        box = ann['bbox']
        size = round(box[2] * box[3], 2)
        size_dict[size].append(ann)

    # Make a copy of the images list for faster lookup
    images_dict = {img['id']: img for img in data['images']}

    for size, anns in tqdm(size_dict.items(), desc='Modifying annotations...'):
        if len(anns) > min_count:
            excess_anns = random.sample(anns, len(anns) - min_count)
            for ann in excess_anns:
                # Add noise to the width and height
                noise_w = np.random.normal(0, noise_stddev)
                noise_h = np.random.normal(0, noise_stddev)
                ann['bbox'][2] += noise_w
                ann['bbox'][3] += noise_h

                # Make sure the box doesn't exceed the image dimensions
                if ann['bbox'][0] + ann['bbox'][2] > width:
                    ann['bbox'][0] = width - ann['bbox'][2]
                if ann['bbox'][1] + ann['bbox'][3] > height:
                    ann['bbox'][1] = height - ann['bbox'][3]

    # Save the modified annotation data
    with open(annotation_file, 'w') as f:
        json.dump(data, f)

    print(f'Downsampling done. The minimum count of each size is now {min_count}.')


@wrp
def balance_sizes(annotation_file, images_dir, multiplier=0.5):
    # Load annotation data
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    # Calculate bounding box sizes and count occurrences
    boxes = np.array([ann['bbox'] for ann in data['annotations']])
    sizes = np.round(boxes[:, 2] * boxes[:, 3], 2)
    size_counts = Counter(sizes)

    # Calculate the mean and std count of bounding box sizes
    mean_count = mean(size_counts.values())
    std_count = stdev(size_counts.values())

    # Calculate the median count of bounding box sizes
    median_count = median(size_counts.values())

    compremize = max((median_count + abs(mean_count - std_count) * (multiplier ** 2)) * (1 + multiplier), 1)
    compremize = int(round(compremize))

    msg(f"Compromising a size count of {compremize}")

    xremove_outliers(image_dir=images_dir, output_dir=images_dir, annotations_file=annotation_file, aggressiveness=2,
                     ratio_bounds=(0.8, 1 / 0.8))
    downsample_sizes(annotation_file=annotation_file, images_dir=images_dir, min_count=compremize)
    xremove_outliers(image_dir=images_dir, output_dir=images_dir, annotations_file=annotation_file, aggressiveness=0,
                     ratio_bounds=(0.8, 1 / 0.8))
    upsample_sizes(annotation_file=annotation_file, images_dir=images_dir, max_count=compremize, max_inplace_dup=1)


@wrp
def cleanup_annotations_and_images(annotation_file, images_dir):
    # Load annotation data
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    print('Removing images without annotations...')
    # Find images without annotations
    image_ids_with_annotations = {ann['image_id'] for ann in data['annotations']}
    images_without_annotations = [img for img in data['images'] if img['id'] not in image_ids_with_annotations]

    # Remove images without annotations from annotation data
    data['images'] = [img for img in data['images'] if img not in images_without_annotations]

    # Delete image files without annotations
    for img in tqdm(images_without_annotations):
        img_path = os.path.join(images_dir, img['file_name'])
        if os.path.exists(img_path):
            print(f"Image: {img_path}")
            os.remove(img_path)

    print('Removing images that do not have corresponding annotations...')
    # Find images that do not have corresponding annotations
    image_files_in_annotations = {img['file_name'] for img in data['images']}
    images_in_dir = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

    # Remove image files that do not have corresponding annotations
    for img_file in tqdm(images_in_dir):
        if img_file not in image_files_in_annotations:
            os.remove(os.path.join(images_dir, img_file))

    # Save the modified annotation data
    with open(annotation_file, 'w') as f:
        json.dump(data, f)

    print('Cleanup done.')


@wrp
def analyze_annotations(input_json_path, plot=False):
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

    data_to_plot = {
        'Bounding Box Sizes': bbox_sizes,
        'Aspect Ratios': aspect_ratios,
        # 'Bounding Box Heights': bbox_heights,
        # 'Bounding Box Widths': bbox_widths,
        'Annotations Per Image': annotations_per_image_counts
    }

    if plot:
        for key, value in data_to_plot.items():
            fig = plt.figure(figsize=(max(5, len(np.unique(value)) // 200), 5))  # dynamic figure size
            plt.hist(value, bins=min(50, len(np.unique(value))))
            plt.title(f'Histogram of {key}')
            plt.xlabel(key)
            plt.ylabel('Count')
            manager = plt.get_current_fig_manager()
            plt.show(block=False)

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

    pprint(analysis_results)

    return analysis_results


@wrp
def convert_bgr_to_rgb_in_dir(directory):
    for filename in q(os.listdir(directory), desc='Converting to RGB ...'):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Add/modify the file extensions as needed.
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_path, img_rgb)


@wrp
def rename_file_inplace(file_path, new_name):
    try:
        directory = os.path.dirname(file_path)
        new_file_path = os.path.join(directory, new_name)
        os.rename(file_path, new_file_path)
    finally:
        return new_file_path


@wrp
def copy_folder(source_folder, destination_folder):
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
    # Copy the folder and its contents
    shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)


def equalize_histogram(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def denoise(img):
    return cv2.fastNlMeansDenoisingColored(img)


def sharpen_image(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 5)
    return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)


def increase_saturation(img, saturation_scale=1.1):
    # Convert BGR image to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float32')

    # Scale the saturation channel
    img_hsv[:, :, 1] = img_hsv[:, :, 1] * saturation_scale

    # Make sure the values stay in the valid range [0, 255]
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)

    # Convert the HSV image back to BGR
    img_bgr = cv2.cvtColor(img_hsv.astype('uint8'), cv2.COLOR_HSV2BGR)
    return img_bgr


def process_image(img, full=True):
    if full:
        img = increase_saturation(img)
        img = equalize_histogram(img)
        img = denoise(img)
    else:
        img = sharpen_image(img)
    return img


def process_and_save(filename, full):
    img = cv2.imread(filename)
    enhanced = process_image(img, full)
    enhanced_filename = filename  # os.path.splitext(filename)[0] + "_enhanced" + os.path.splitext(filename)[1]
    cv2.imwrite(enhanced_filename, enhanced)


@wrp
def enhance_images(directory, full):
    filenames = [os.path.join(directory, f) for f in os.listdir(directory) if
                 (f.endswith(".jpg") or f.endswith(".png")) and not f[:-4] in BAD_IMGS]

    # with Pool() as p:
    #     list(tqdm(p.imap(process_and_save, filenames), total=len(filenames), desc="Processing images"))

    for filename in tqdm(filenames, desc="Processing images"):
        if not filename[:-4] in BAD_IMGS:
            process_and_save(filename, full)


@wrp
def remove_images_and_annotations(images_to_remove, annotations_file, images_directory):
    # Load the annotations file
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    # Create a regular expression pattern to match variations of the file names
    patterns = [re.compile(r'^{}(_\w+)*'.format(re.escape(image_name))) for image_name in images_to_remove]
    pprint(patterns)

    # Get the IDs of the images to remove
    image_ids_to_remove = [image['id'] for image in data['images'] if
                           any(pattern.match(image['file_name']) for pattern in patterns)]
    pprint(image_ids_to_remove)
    msg(f"Removing: {len(image_ids_to_remove)} / {len(data['images'])}")

    # Remove the images from the 'images' field
    data['images'] = [image for image in data['images'] if image['id'] not in image_ids_to_remove]

    # Remove the annotations of the images from the 'annotations' field
    data['annotations'] = [annotation for annotation in data['annotations'] if
                           annotation['image_id'] not in image_ids_to_remove]

    # Remove the image files
    for image_name in images_to_remove:
        image_path = os.path.join(images_directory, image_name)
        if os.path.exists(image_path):
            os.remove(image_path)

    # Save the modified data back to the file
    with open(annotations_file, 'w') as f:
        json.dump(data, f)


@wrp
def rename_images_and_update_annotations(annotations_file, images_directory):
    # Load the annotations file
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    # Dictionary to store new filenames
    new_filenames = {}

    # Rename the image files and store new names
    for image in q(data['images']):
        old_name = image['file_name']
        ext = os.path.splitext(old_name)[-1]
        new_name = str(uuid.uuid4()).replace('-', '') + ext
        os.rename(os.path.join(images_directory, old_name), os.path.join(images_directory, new_name))
        new_filenames[old_name] = new_name

    # Update the annotations with the new filenames
    for image in q(data['images']):
        old_name = image['file_name']
        image['file_name'] = new_filenames[old_name]

    # Save the modified data back to the file
    with open(annotations_file, 'w') as f:
        json.dump(data, f)


def create_folders(directories):
    for d in directories:
        os.makedirs(d, exist_ok=True)


def create_files(files):
    for f in files:
        with open(f, 'w') as ff:
            ff.close()


@wrp
def process(root_dir, version, num_images=4, num_augmented_images=4):
    json_file = r"\annotations.json"
    borig_dir = root_dir + r"\train"  # r"\train"  #
    borig_json = borig_dir + r"\_annotations.coco.json"
    orig_dir = root_dir + r"\train_data"  # r"\train"  #
    orig_json = orig_dir + r"\_annotations.coco.json"
    ovr_dir = f"{root_dir}\ovr_{version}"
    ovr_json = ovr_dir + json_file
    div_dir = f"{root_dir}\div_{version}"
    div_json = div_dir + json_file
    blc_dir = root_dir + rf"\blc_{version}"
    blc_json = blc_dir + json_file
    blc_sjson = blc_dir + rf"\sc_annotations.json"
    aug_dir = root_dir + rf"\aug_{version}"
    aug_json = aug_dir + json_file
    chegg_dir = root_dir + rf"\chegg"
    chegg_json = chegg_dir + rf"/sc_annotations.json"

    copy_folder(borig_dir, orig_dir)
    create_folders((div_dir, blc_dir, aug_dir))
    create_files((div_json, blc_json, aug_json))

    orig_json = rename_file_inplace(orig_json, json_file[1:])

    # display_random_image_with_boxes(orig_json)
    # analyze_annotations(orig_json, plot=True)

    fix_coco_annotations(orig_json)

    # remove_abnormal_images(orig_json, orig_dir, std_multiplier=0.25)
    enhance_images(orig_dir, True)
    split_and_overlap_dataset(orig_json, orig_dir, div_dir, num_images, 1/3)
    remove_images_and_annotations(BAD_IMGS, div_json, div_dir)
    enhance_images(div_dir, False)
    remove_edge_annotations(div_json)
    remove_overlapping_annotations(div_json, threshold=1/3)
    # xremove_outliers(image_dir=div_dir, output_dir=div_dir, annotations_file=div_json, aggressiveness=2.0,
    #                  ratio_bounds=(0.8, 1 / 0.8))
    # display_random_image_with_boxes(div_json)
    # analyze_annotations(div_json, plot=True)

    # fix_size_outliers(orig_json)
    # display_random_image_with_boxes(orig_json)
    # analyze_annotations(orig_json, plot=True)
    # display_random_image_with_boxes(orig_json)
    # analyze_annotations(orig_json, plot=True)
    augment_dataset(div_dir, div_json, aug_dir, num_augmented_images, True)
    # analyze_annotations(aug_json, plot=True)
    # display_random_image_with_boxes(aug_json)

    check_image_sizes(aug_json, aug_dir)
    # cleanup_annotations_and_images(div_json, div_dir)
    # xremove_outliers(image_dir=aug_dir, output_dir=aug_dir, annotations_file=aug_json, aggressiveness=2.5,
    #                  ratio_bounds=(0.8, (1 / 0.8)), rm_overlap=True)

    # display_random_image_with_boxes(aug_json)

    undersample_coco(aug_json, aug_dir, blc_dir, blc_json)

    # analyze_annotations(blc_json, plot=True)
    # fix_size_outliers(blc_json)
    balance_sizes(blc_json, blc_dir, 1 / math.sqrt(2))
    # check_image_sizes(blc_json, blc_dir)
    # cleanup_annotations_and_images(blc_json, blc_dir)
    coco_to_single_class(blc_json, blc_sjson)
    analyze_annotations(blc_json, plot=True)
    msg(f"{get_optimal_anchors(blc_json)}")
    analyze_coco_annotations(blc_json)
    # display_random_image_with_boxes(blc_json)
    # analyze_coco_annotations(blc_sjson)
    # calc_mean_std(blc_json)

    # analyze_coco_annotations(blc_sjson)
    # analyze_annotations(blc_sjson)
    # analyze_coco_annotations(blc_json)
    # analyze_annotations(blc_json)


root_dir = r"C:\Users\LTSS2023\Documents\elhadjmb\datasets"
# root_dir = r"D:\TMP"

process(root_dir=root_dir, version=55, num_images=9, num_augmented_images=8)
