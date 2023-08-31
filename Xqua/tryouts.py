import copy
import json
import math
import os
import random
import shutil
from collections import defaultdict, Counter
from pathlib import Path
from pprint import pprint
from typing import List

import json
import os
from PIL import Image, ImageFilter, ImageFile
from tqdm import tqdm
from collections import Counter
import numpy as np
import random
from joblib import Parallel, delayed
from rtree import index
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from scipy.stats import iqr
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
from tqdm.auto import tqdm
from collections import Counter
from joblib import Parallel, delayed
from tqdm import tqdm as q


def display_random_image_with_boxes(json_file):
    # Define a list of predefined colors for different classes
    class_colors = ['c', 'r', 'g', 'b', 'm', 'y']

    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    while True:
        # Select a random image from the dataset
        image_info = random.choice(data['images'])
        image_path = os.path.join(os.path.dirname(json_file), image_info['file_name'])

        # Find the corresponding annotations for the selected image
        annotations = [anno for anno in data['annotations'] if anno['image_id'] == image_info['id']]
        if len(annotations) > 1:
            break

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
        print("Failed to load the image:", image_path)


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
        aspect_ratios.append(float(width) / height) if height != 0 else 999999999
        bbox_heights.append(height)
        bbox_widths.append(width)

    bbox_sizes = np.array(bbox_sizes)
    aspect_ratios = np.array(aspect_ratios)
    bbox_heights = np.array(bbox_heights)
    bbox_widths = np.array(bbox_widths)

    if plot:
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
    pprint(analysis_results)
    return analysis_results


def remove_images_and_annotations(images_to_remove, annotations_file):
    def get_filename(file_path):
        return os.path.basename(file_path)

    try:
        # Load the annotations file
        with open(annotations_file, 'r') as f:
            data = json.load(f)

        # Get the filenames of the images to remove
        filenames_to_remove = set([get_filename(image['file_name']) for image in data['images']])
        filtered_images = [image for image in data['images'] if
                           get_filename(image['file_name']) not in images_to_remove]

        # Remove the images from the 'images' field
        data['images'] = filtered_images

        # Remove the annotations of the images from the 'annotations' field
        filtered_annotations = [annotation for annotation in data['annotations'] if
                                get_filename(
                                    data['images'][annotation['image_id']]['file_name']) not in filenames_to_remove]
        data['annotations'] = filtered_annotations

        # Save the modified data back to the file
        with open(annotations_file, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print("An error occurred:", e)


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
    for image_id, annos in annotations_per_image.items():
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

    return anchor_sizes, anchor_ratios


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


def remove_outliers(annotations_file, image_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load the annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    # Create a lookup dictionary for image filenames
    image_lookup = {image['id']: image['file_name'] for image in annotations['images']}

    # Calculate aspect ratios for all bounding boxes
    aspect_ratios = []
    for annotation in tqdm(annotations['annotations']):
        bbox = annotation['bbox']
        aspect_ratio = bbox[2] / bbox[3]
        aspect_ratios.append(aspect_ratio)

    # Calculate the first quartile, third quartile, and the IQR of the aspect ratios
    q1 = np.percentile(aspect_ratios, 25)
    q3 = np.percentile(aspect_ratios, 75)
    iqr = q3 - q1

    # Define the upper and lower bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Iterate over the annotations
    for annotation in tqdm(annotations['annotations']):
        # If the aspect ratio is an outlier, remove the annotation and draw over the bounding box in the image
        aspect_ratio = annotation['bbox'][2] / annotation['bbox'][3]
        if aspect_ratio < lower_bound or aspect_ratio > upper_bound:
            # Load the image
            image_file = os.path.join(image_dir, image_lookup[annotation['image_id']])
            image = cv2.imread(image_file)

            # Draw over the bounding box in black
            x, y, w, h = [int(coord) for coord in annotation['bbox']]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)

            # Save the modified image
            output_file = os.path.join(output_dir, image_lookup[annotation['image_id']])
            cv2.imwrite(output_file, image)

            # Remove the annotation
            annotations['annotations'].remove(annotation)

    # Save the modified annotations in the new images directory
    output_annotations_file = os.path.join(output_dir, 'annotations.json')
    with open(output_annotations_file, 'w') as f:
        json.dump(annotations, f)


def remove_outliers_with_isolation_forest(annotations_file, image_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(annotations_file) as f:
        data = json.load(f)

    # Extract width and height of bounding boxes
    widths = []
    heights = []
    for annotation in data['annotations']:
        bbox = annotation['bbox']
        widths.append(bbox[2])
        heights.append(bbox[3])

    # Calculate areas and aspect ratios
    areas = np.array(widths) * np.array(heights)
    aspect_ratios = np.array(widths) / np.array(heights)

    # Fit Isolation Forest
    X = np.column_stack((areas, aspect_ratios))
    clf = IsolationForest(contamination=0.01)
    clf.fit(X)
    preds = clf.predict(X)

    # Identify outliers
    outliers = np.where(preds == -1)[0]

    # Remove outliers from annotations and draw over them in images
    new_annotations = [a for i, a in enumerate(data['annotations']) if i not in outliers]
    data['annotations'] = new_annotations

    # Create a dictionary for quick lookup of image id to file name
    id_to_filename = {image['id']: image['file_name'] for image in data['images']}

    for outlier in tqdm(outliers):
        image_id = data['annotations'][outlier]['image_id']
        filename = id_to_filename[image_id]
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        bbox = data['annotations'][outlier]['bbox']
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
        new_image_path = os.path.join(output_dir, filename)
        cv2.imwrite(new_image_path, image)

    # Save the modified annotations file in the new images directory
    with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
        json.dump(data, f)


def remove_outliersx(annotations_file, image_dir, output_dir):
    with open(annotations_file) as f:
        data = json.load(f)

    # Create output directories if they don't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'images')):
        os.makedirs(os.path.join(output_dir, 'images'))

    # Create a mapping from image id to file name
    id_to_filename = {image['id']: image['file_name'] for image in data['images']}

    # Calculate areas and aspect ratios of bounding boxes
    areas = []
    ratios = []
    for annotation in data['annotations']:
        bbox = annotation['bbox']
        area = bbox[2] * bbox[3]
        ratio = bbox[2] / bbox[3]
        areas.append(area)
        ratios.append(ratio)

    # Convert to numpy arrays
    areas = np.array(areas).reshape(-1, 1)
    ratios = np.array(ratios).reshape(-1, 1)

    # Apply Isolation Forest and IQR for both areas and ratios
    for feature in tqdm([areas, ratios]):
        # Apply Isolation Forest
        clf = IsolationForest(contamination=0.01)
        clf.fit(feature)
        isof_outliers = clf.predict(feature) == -1

        # Apply IQR
        Q1 = np.percentile(feature[~isof_outliers], 25)
        Q3 = np.percentile(feature[~isof_outliers], 75)
        IQR = iqr(feature[~isof_outliers])
        iqr_outliers = (feature < (Q1 - 1.5 * IQR)) | (feature > (Q3 + 1.5 * IQR))

        # Combine outliers
        outliers = isof_outliers | iqr_outliers.flatten()

        # Remove outlier annotations and draw over their bounding boxes in images
        new_annotations = []
        modified_images = {}
        for i, annotation in enumerate(data['annotations']):
            if outliers[i]:
                image_id = annotation['image_id']
                image_path = os.path.join(image_dir, id_to_filename[image_id])
                if image_id not in modified_images:
                    modified_images[image_id] = cv2.imread(image_path)
                image = modified_images[image_id]
                bbox = annotation['bbox']
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                              (0, 0, 0), -1)
            else:
                new_annotations.append(annotation)

        # Save modified images
        for image_id, image in tqdm(modified_images.items()):
            cv2.imwrite(os.path.join(output_dir, 'images', id_to_filename[image_id]), image)

        # Save new annotations
        data['annotations'] = new_annotations
        with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
            json.dump(data, f)


def remove_outliersy(annotations_file, image_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

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
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return lower_bound, upper_bound

    # Get the bounds for aspect ratios and sizes
    aspect_ratio_bounds = get_bounds(aspect_ratios)
    size_bounds = get_bounds(sizes)

    # Create a dictionary to keep track of modified images
    modified_images = {}

    # Iterate over the annotations
    for annotation in tqdm(annotations['annotations']):
        # If the aspect ratio or size is an outlier, remove the annotation and draw over the bounding box in the image
        aspect_ratio = annotation['bbox'][2] / annotation['bbox'][3]
        size = annotation['bbox'][2] * annotation['bbox'][3]
        if (aspect_ratio < aspect_ratio_bounds[0] or aspect_ratio > aspect_ratio_bounds[1] or
                size < size_bounds[0] or size > size_bounds[1]):
            # Load the image if it hasn't been loaded yet
            image_id = annotation['image_id']
            if image_id not in modified_images:
                image_file = os.path.join(image_dir, image_lookup[image_id])
                modified_images[image_id] = cv2.imread(image_file)

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

            # Remove the annotation
            annotations['annotations'].remove(annotation)

    # Save all modified images
    for image_id, image in modified_images.items():
        output_file = os.path.join(output_dir, image_lookup[image_id])
        cv2.imwrite(output_file, image)

    # Save the modified annotations in the new images directory
    output_annotations_file = os.path.join(output_dir, 'annotations.json')
    with open(output_annotations_file, 'w') as f:
        json.dump(annotations, f)


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


def remove_outliersz(annotations_file, image_dir, output_dir, aggressiveness=1.0, ratio_bounds=None):
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


import json
import numpy as np
import math
from scipy.stats import iqr


def freedman_diaconis(data):
    """Determine number of bins using Freedman-Diaconis rule."""
    data_range = np.ptp(data)
    bin_width = 2 * iqr(data) / np.power(len(data), 1 / 3)
    return int(np.ceil(data_range / bin_width))


def fix_size_outliers(annotation_file):
    # Load annotation data
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    sizes = []
    aspect_ratios = []
    for annotation in tqdm(data['annotations']):
        bbox = annotation['bbox']
        width = bbox[2]
        height = bbox[3]
        sizes.append(width * height)
        aspect_ratios.append(width / height)

    # Convert to numpy arrays for easier manipulation
    sizes = np.array(sizes)
    aspect_ratios = np.array(aspect_ratios)

    # Calculate number of bins dynamically using Freedman-Diaconis rule
    n_bins = int(freedman_diaconis(sizes) * 1.5)  # added 50% more the number of bins for extra flexibility

    print(f"Number of bins: {n_bins} ({n_bins / 1.5})")
    # Bin indices for each element in sizes
    bin_indices = np.digitize(sizes, bins=np.histogram(sizes, bins=n_bins)[1])

    # Adjust sizes
    for bin_i in tqdm(range(1, n_bins + 1)):
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


def do():
    root = r"C:\Users\mirt0\Downloads\train/"
    annotations_file = f"{root}annotations.json"
    outt = f"{root}outQQ/"
    outtj = f"{outt}annotations.json"

    # e = analyze_annotations(annotations_file)
    # pprint(e)
    #
    # pprint(suggest_anchors(analyze_annotations(annotations_file)))
    #
    # pprint(get_optimal_anchors(annotations_file))
    #
    msg('removing outliers')
    remove_outliersz(annotations_file=annotations_file, image_dir=root, output_dir=outt, aggressiveness=2.0,
                     ratio_bounds=(0.6666, 1 / 0.6666))
    msg("After")
    pprint(get_optimal_anchors(outtj))

    e = analyze_annotations(outtj, True)
    pprint(e)

    for _ in range(8):
        display_random_image_with_boxes(outtj)


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


def xremove_outliers(annotations_file, image_dir, output_dir, aggressiveness=1.0, ratio_bounds=None, rm_overlap=True):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if rm_overlap:
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


import cv2
import numpy as np
import random
import os
import json
from multiprocessing import Pool


def calculate_edge_average_color(image, bbox):
    xmin, ymin, width, height = [int(coord) for coord in bbox]
    xmax, ymax = xmin + width, ymin + height

    edge_pixels = np.concatenate((
        image[ymin, xmin:xmax],
        image[ymax, xmin:xmax],
        image[ymin:ymax, xmin],
        image[ymin:ymax, xmax]
    ))

    return [int(channel) for channel in np.mean(edge_pixels, axis=0)]


def modify_image(args):
    image_path, annotations_to_remove = args
    image = cv2.imread(image_path)

    for annotation in annotations_to_remove:
        bbox = annotation['bbox']
        avg_color = calculate_edge_average_color(image, bbox)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), avg_color, -1)

    cv2.imwrite(image_path, image)


def process_sizes(size, max_count, size_dict):
    # Upscale each size in a separate process
    if len(size_dict[size]) < max_count:
        duplicates = random.choices(size_dict[size], k=max_count - len(size_dict[size]))
        return duplicates
    else:
        return []


def draw_over_annotations(img, ann):
    # Get bounding box coordinates
    x, y, w, h = [int(coord) for coord in ann['bbox']]
    # Ensure the indices are within the image dimensions
    height, width, _ = img.shape
    x = min(x, width - 1)
    y = min(y, height - 1)
    w = min(w, width - x)
    h = min(h, height - y)
    # Get the border pixels of the bounding box
    top_border = img[y, x:x + w]
    bottom_border = img[min(y + h, height - 1), x:x + w]
    left_border = img[y:y + h, x]
    right_border = img[y:y + h, min(x + w, width - 1)]
    # Calculate the average color of the border pixels
    border_pixels = np.concatenate((top_border, bottom_border, left_border, right_border), axis=0)
    avg_color = np.mean(border_pixels, axis=0)
    # Draw over the bounding box with the average color
    cv2.rectangle(img, (x, y), (x + w, y + h), tuple(avg_color), -1)

    return img


def upsample_sizes(annotation_file, images_dir, max_count, image_dup_thresh=5, n_jobs=-1):
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

    print("Stage 1: Duplicate entire images along with relevant annotations")
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

    print("Stage 2: Duplicate annotations in the same images")
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

    print(f'Upsampling done. The maximum count of each size is now {max_count}.')


def downsample_sizes(annotation_file, images_dir, min_count):
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

    print(f'Downsampling done. The minimum count of each size is now {min_count}.')


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

    compremize = (median_count + mean_count + std_count) * (1 + multiplier)
    compremize = int(round(compremize))

    print(f"Comprimizing size of {compremize}")
    downsample_sizes(annotation_file=annotation_file, images_dir=images_dir, min_count=compremize)
    upsample_sizes(annotation_file=annotation_file, images_dir=images_dir, max_count=compremize, image_dup_thresh=5)


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

    print(f'Removed {len(ids_to_remove)} images from annotations.')


def de():
    root = r"C:\Users\mirt0\Downloads\train/"
    annotations_file = root + r"_annotations.coco.json"
    div = root + "div/"
    diva = div + "annotations.json"
    # undersample_coco(input_json_path=annotations_file, images_dir, output_dir, output_json_path)
    for _ in range(5):
        display_random_image_with_boxes(diva)


def da():
    pass


import cv2
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm


def equalize_histogram(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def denoise(img):
    return cv2.fastNlMeansDenoisingColored(img)


def sharpen_image(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 5)
    return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)


def process_image(img):
    img = equalize_histogram(img)
    img = denoise(img)
    img = sharpen_image(img)
    return img


def process_and_save(filename):
    img = cv2.imread(filename)
    enhanced = process_image(img)
    enhanced_filename = filename  # os.path.splitext(filename)[0] + "_enhanced" + os.path.splitext(filename)[1]
    cv2.imwrite(enhanced_filename, enhanced)


def enhance_images(directory):
    filenames = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".jpg") or f.endswith(".png")]
    with Pool() as p:
        list(tqdm(p.imap(process_and_save, filenames), total=len(filenames), desc="Processing images"))


def rename_file_inplace(file_path, new_name):
    try:
        directory = os.path.dirname(file_path)
        new_file_path = os.path.join(directory, new_name)
        os.rename(file_path, new_file_path)
    finally:
        return new_file_path


if __name__ == '__main__':
    # do()
    rrot = r"C:\Users\mirt0\Downloads\train"
    jsn = rename_file_inplace(r"C:\Users\mirt0\Downloads\train/_annotations.coco.json", "annotations.json")
    analyze_annotations(r"D:\\TMP\\blc_123\\annotations.json", True)
    # enhance_images(r"D:\\TMP\\train")
    # remove_abnormal_images(jsn, rrot, std_multiplier=0.2)
    # fix_size_outliers(jsn)
    # xremove_outliers(jsn, rrot, rrot, 2.0, (0.8333, (1 / 0.8333)), True)
    # analyze_annotations(jsn, True)
    # balance_sizes(jsn, rrot, 1 / math.sqrt(2))
    # analyze_annotations(jsn, True)
    #
    # print(get_optimal_anchors(jsn))
    # fix_coco_annotations(jsn)
    # for _ in range(5):
    #     display_random_image_with_boxes(jsn)

    lst = [
        r"IMG20230418125507_001_jpg.rf.17d5829a9cc38db9c7ac26618fcb0b5d.jpg",
        r"IMG20230418125507_001_jpg.rf.c35e080bf89ee1b9aaeb9f0f68e8a966.jpg",
        r"IMG20230418125507_001_jpg.rf.c68b9a83175b7b8de42bc213dfc86a56.jpg",
        r"IMG20230418125507_002_jpg.rf.5ef0e57dcf1c6c531feb48fd25d11dae.jpg",
        r"IMG20230418125507_002_jpg.rf.53910f331b5996d25e960a52707e6fbc.jpg",
        r"IMG20230418125507_002_jpg.rf.537794e5b232c84fc1b9bab181523d22.jpg",
        r"IMG20230418125507_004_jpg.rf.3b9954965ae7bc6b1fa5edaacd73c3e2.jpg",
        r"IMG20230418125507_004_jpg.rf.a07d3f1c95a0a9e44542f1e5d829442c.jpg",
        r"IMG20230418125507_004_jpg.rf.f8cbd2cc874b24a1a525400acb7cfc24.jpg",

    ]

    remove_images_and_annotations(lst, r"D:\\TMP\\train\\_annotations.coco.json")
