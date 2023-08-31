import json
import math
import os
import shutil
from collections import Counter
from collections import defaultdict
from pathlib import Path
from random import choices
from random import sample
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from matplotlib import ticker
from rtree import index
from sklearn.cluster import KMeans
from tqdm import tqdm

# from tr7_clb_bkp import msg, wrp


@wrp
def split_dataset(annotation_path, img_dir, output_dir, number_of_imgs):
    msg(
        f"Running split_dataset with:\nannotation_path: {annotation_path}\nimg_dir: {img_dir}\noutput_dir: {output_dir}\nF: {number_of_imgs}")
    # Read the annotations
    with open(annotation_path) as f:
        annotations = json.load(f)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Determine the horizontal and vertical split count based on F (assuming F is a square number)
    split_count = int(math.sqrt(number_of_imgs))
    msg(
        f"Each image will be split into {split_count}x{split_count} segments.")

    # Estimate average object size
    total_width = total_height = 0
    count = 0
    for annotation in annotations['annotations']:
        x, y, w, h = annotation['bbox']
        total_width += w
        total_height += h
        count += 1
    avg_width = total_width / count
    avg_height = total_height / count

    new_annotations = dict()
    new_images_info = []
    new_annotations_list = []

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

        # Calculate overlap
        overlap_width = int(max(avg_width, new_width * 0.2))
        overlap_height = int(max(avg_height, new_height * 0.2))

        # Split the image and adjust the annotations
        new_images = []
        for i in range(split_count):
            for j in range(split_count):
                # Extract the image segment
                new_image = img[
                            max(0, i * new_height - overlap_height):min(height, (i + 1) * new_height + overlap_height),
                            max(0, j * new_width - overlap_width):min(width, (j + 1) * new_width + overlap_width)]
                new_images.append(new_image)

                # Adjust the annotations
                for annotation in annotations['annotations']:
                    if annotation['image_id'] == image_id:
                        x, y, w, h = annotation['bbox']
                        if max(0, j * new_width - overlap_width) <= x + w / 2 <= min(width, (
                                                                                                    j + 1) * new_width + overlap_width) and max(
                            0, i * new_height - overlap_height) <= y + h / 2 <= min(height, (
                                                                                                    i + 1) * new_height + overlap_height):
                            new_annotation = annotation.copy()
                            new_annotation['bbox'] = [
                                x - max(0, j * new_width - overlap_width), y - max(0, i * new_height - overlap_height),
                                w, h]
                            new_annotation['image_id'] = len(
                                new_images_info) + len(new_images) - 1
                            # Update the annotation ID here
                            new_annotation['id'] = len(
                                new_annotations_list)
                            new_annotations_list.append(new_annotation)

        # Save the new images and update the image info list
        for k, new_image in enumerate(new_images):
            new_file_name = file_name.replace('.jpg', f'_{k}.jpg')
            cv2.imwrite(output_dir + '/' + new_file_name, new_image)
            new_id = image_id * number_of_imgs + k
            new_images_info.append(
                {'id': new_id, 'file_name': new_file_name, 'width': new_width, 'height': new_height})

    # After all images and annotations have been processed...
    image_ids = set([image['id'] for image in new_images_info])
    # We're now checking annotation IDs, not annotation image IDs
    annotation_ids = set([annotation['id']
                          for annotation in new_annotations_list])
    assert len(image_ids) == len(
        new_images_info), "Duplicate image ids detected!"
    assert len(annotation_ids) == len(
        new_annotations_list), "Duplicate annotation ids detected!"

    new_annotations = annotations
    new_annotations['images'] = new_images_info
    new_annotations['annotations'] = new_annotations_list

    # Write the new annotations
    with open(output_dir + '/annotations.json', 'w') as number_of_imgs:
        json.dump(new_annotations, number_of_imgs)

    msg(
        f"Saved {len(new_annotations_list)} new annotations and the updated annotations to {output_dir}/annotations.json.")


@wrp
def augment_dataset(images_dir, annotation_file, output_dir, num_augmented_images=25, simple=True):
    # Load the annotation file
    with open(annotation_file, 'r') as f:
        original_annotations = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Define the categories
    color_augmenters = [
        iaa.Add((-10, 10)),
        iaa.Multiply((0.8, 1.2)),
        iaa.AddToHueAndSaturation((-5, 5)),
        iaa.GammaContrast((0.7, 1.3), per_channel=True)
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
        iaa.geometric.Affine(rotate=(-90, 90), shear=(10, 10), mode='reflect')
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
        iaa.Sometimes(np.random.uniform(0.4, 0.7), iaa.SomeOf((1, 2), edge_augmenters)),
        iaa.Sometimes(1, iaa.SomeOf((1, 2), flip_augmenters)),
        iaa.Sometimes(np.random.uniform(0, 0.3), iaa.OneOf(color_augmenters)),
        iaa.Sometimes(np.random.uniform(0, 0.5), iaa.OneOf(weather_augmenters)),
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
def undersample_coco(input_json_path, images_dir, output_dir, output_json_path):
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

    # Convert selected_annotations to a set for faster membership checking
    selected_annotations = set(selected_annotations)

    msg(
        "Updating the images list to include only images with selected annotations ... (this takes a lot of time please wait)")
    selected_images = [img for img in tqdm(images) if any(
        anno['image_id'] == img['id'] for anno in selected_annotations)]

    msg("Copying the selected images to the output directory")
    for img in tqdm(selected_images):
        shutil.copy(os.path.join(images_dir, img['file_name']), os.path.join(
            output_dir, img['file_name']))

    # Create a dictionary to keep track of modified images
    modified_images = {}

    # Sort annotations by image ID so that all annotations for the same image are processed consecutively
    annotations.sort(key=lambda anno: anno['image_id'])

    # Iterate over all annotations
    for annotation in tqdm(annotations):
        # Load the image if it hasn't been loaded yet
        image_id = annotation['image_id']
        if image_id not in modified_images:
            image_file = os.path.join(images_dir, image_lookup[image_id])
            modified_images[image_id] = cv2.imread(image_file)

        # If the annotation is not selected, draw over it
        if annotation not in selected_annotations:
            # Get the bounding box coordinates
            x, y, w, h = [int(coord) for coord in annotation['bbox']]

            # Ensure the indices are within the image dimensions
            height, width, _ = modified_images[image_id].shape
            x = min(x, width - 1)
            y = min(y, height - 1)
            w = min(w, width - x)
            h = min(h, height - y)

            # Get the border pixels of the bounding box
            top_border = modified_images[image_id][y, x:x + w]
            bottom_border = modified_images[image_id][min(y + h, height - 1), x:x + w]
            left_border = modified_images[image_id][y:y + h, x]
            right_border = modified_images[image_id][y:y + h, min(x + w, width - 1)]

            # Calculate the average color of the border pixels
            border_pixels = np.concatenate((top_border, bottom_border, left_border, right_border), axis=0)
            avg_color = np.mean(border_pixels, axis=0)

            # Draw over the bounding box with the average color
            cv2.rectangle(modified_images[image_id], (x, y), (x + w, y + h), avg_color, -1)

    # Save all modified images
    for image_id, image in modified_images.items():
        output_file = os.path.join(output_dir, image_lookup[image_id])
        cv2.imwrite(output_file, image)

    # Update the annotations list to include only selected annotations
    annotations = [anno for anno in annotations if anno in selected_annotations]

    annotation_dict = {'info': data['info'], 'licenses': data['licenses'],
                       'images': selected_images, 'annotations': annotations, 'categories': categories}
    msg("Saving the new annotations file")
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


@wrp
def process(root_dir, version, num_images=4, num_augmented_images=2):
    json_file = r"\annotations.json"
    orig_dir = root_dir + r"\fixed"  # r"\original"
    orig_json = orig_dir + json_file
    ovr_dir = f"{root_dir}\ovr_{version}"
    ovr_json = ovr_dir + json_file
    div_dir = f"{root_dir}\div_{version}"
    div_json = div_dir + json_file
    blc_dir = root_dir + rf"\blc_{version}"
    blc_json = blc_dir + json_file
    blc_sjson = blc_dir + rf"\sc_annotations.json"
    aug_dir = root_dir + rf"\aug_{version}"
    aug_json = aug_dir + json_file

    fix_coco_annotations(orig_json)
    remove_outliers(image_dir=orig_dir, output_dir=orig_dir, annotations_file=orig_json, aggressiveness=2.0,
                    ratio_bounds=(0.5, 2.0))
    split_dataset(orig_json, orig_dir, div_dir, num_images)
    # # #
    # # # oversample_coco(div_json, div_dir, ovr_dir, ovr_json)
    # #
    # # clean_annotations_until_threshold(div_json, max_data_loss=0.2)
    #
    augment_dataset(div_dir, div_json, aug_dir, num_augmented_images, True)
    #
    # clean_annotations_until_threshold(aug_json, max_data_loss=0.2)
    remove_outliers(image_dir=div_dir, output_dir=orig_dir, annotations_file=div_json, aggressiveness=2.0,
                    ratio_bounds=(0.5, 2.0))
    undersample_coco(aug_json, aug_dir, blc_dir, blc_json)
    #
    # try:
    #     clean_annotations_until_threshold(blc_json, max_data_loss=0.2)
    # except:
    #     msg("NO 2nd CLEAN")
    #     pass

    coco_to_single_class(blc_json, blc_sjson)

    suggest_anchors(analyze_annotations(blc_json, plot=True))
    msg(f"{get_optimal_anchors(blc_sjson)}")

    analyze_coco_annotations(blc_sjson)
    analyze_coco_annotations(blc_json)
    # calc_mean_std(blc_json)

    # analyze_coco_annotations(blc_sjson)
    # analyze_annotations(blc_sjson)
    # analyze_coco_annotations(blc_json)
    # analyze_annotations(blc_json)


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
def remove_outliers(annotations_file, image_dir, output_dir, aggressiveness=1.0, ratio_bounds=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    remove_overlapping_annotations(annotations_file, threshold=0.5)

    msg(
        f"Running remove_outliersz with:\nannotations_file: {annotations_file}\nimage_dir: {image_dir}\noutput_dir: {output_dir}\naggressiveness: {aggressiveness}\nratio_bounds: {ratio_bounds}")

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
        aspect_ratio = bbox[2] / bbox[3]
        size = bbox[2] * bbox[3]
        aspect_ratios.append(aspect_ratio)
        sizes.append(size)

    # Define a function to calculate the upper and lower bounds for outliers
    def get_bounds(values):
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr * aggressiveness
        upper_bound = q3 + 1.5 * iqr * aggressiveness
        return lower_bound, upper_bound

    # Get the bounds for sizes
    size_bounds = get_bounds(sizes)

    # Create a dictionary to keep track of modified images and their outliers
    modified_images = {}
    outliers = defaultdict(list)

    # Initialize the standard deviation for aspect ratios
    std_dev = np.std(aspect_ratios)

    while True:
        # Get the bounds for aspect ratios
        aspect_ratio_bounds = get_bounds(aspect_ratios) if ratio_bounds is None else ratio_bounds

        # Initialize the lists for the new aspect ratios and sizes
        new_aspect_ratios = []
        new_sizes = []

        # Initialize the lists for the new aspect ratios, sizes, and annotations
        new_aspect_ratios = []
        new_sizes = []
        new_annotations = []

        # Iterate over the annotations
        for annotation in tqdm(annotations['annotations']):
            # If the aspect ratio or size is an outlier, remove the annotation and add it to the outliers
            aspect_ratio = annotation['bbox'][2] / annotation['bbox'][3]
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

        # Check if the standard deviation of the aspect ratios has decreased significantly
        new_std_dev = np.std(new_aspect_ratios)
        if new_std_dev >= std_dev * 0.95:  # Stop if the standard deviation has not decreased by at least 5%
            break

        # Update the aspect ratios, sizes, and standard deviation
        aspect_ratios = new_aspect_ratios
        sizes = new_sizes
        std_dev = new_std_dev

    # Draw over the outliers in the modified images
    for image_id, image_outliers in outliers.items():
        # Load the image if it hasn't been loaded yet
        if image_id not in modified_images:
            image_file = os.path.join(image_dir, image_lookup[image_id])
            modified_images[image_id] = cv2.imread(image_file)

        # Draw over each outlier in the image
        for outlier in image_outliers:
            # Get the bounding box coordinates
            x, y, w, h = [int(coord) for coord in outlier['bbox']]

            # Ensure the indices are within the image dimensions
            height, width, _ = modified_images[image_id].shape
            x = min(x, width - 1)
            y = min(y, height - 1)
            w = min(w, width - x)
            h = min(h, height - y)

            # Get the border pixels of the bounding box
            top_border = modified_images[image_id][y, x:x + w]
            bottom_border = modified_images[image_id][min(y + h, height - 1), x:x + w]
            left_border = modified_images[image_id][y:y + h, x]
            right_border = modified_images[image_id][y:y + h, min(x + w, width - 1)]

            # Calculate the average color of the border pixels
            border_pixels = np.concatenate((top_border, bottom_border, left_border, right_border), axis=0)
            avg_color = np.mean(border_pixels, axis=0)
            # avg_color = (0,0,0)

            # Draw over the bounding box with the average color
            cv2.rectangle(modified_images[image_id], (x, y), (x + w, y + h), avg_color, -1)

    # Save all modified images
    for image_id, image in modified_images.items():
        output_file = os.path.join(output_dir, image_lookup[image_id])
        cv2.imwrite(output_file, image)

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
    for i, annotation in tqdm(enumerate(data['annotations'])):
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


root_dir = r"C:\Users\LTSS2023\Documents\elhadjmb\datasets"

process(root_dir=root_dir, version=10, num_images=4, num_augmented_images=10)
