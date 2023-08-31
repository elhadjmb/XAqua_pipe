import math
import os
import random
from pathlib import Path

import cv2
import imagehash
from PIL import Image
from tqdm import tqdm
from tqdm import tqdm as q


def _check_duplicate(image_path, image_hashes):
    image = Image.open(image_path, image_hashes)
    hashes = {
        "ahash" + str(imagehash.average_hash(image)),
        # "dhash"+str(imagehash.dhash(image)),
        "phash" + str(imagehash.phash(image)),
        "chash" + str(imagehash.colorhash(image)),
        # "whash"+str(imagehash.whash(image)),
    }
    for image_hash in hashes:
        # If the hash already exists in the dictionary, delete the image
        if image_hash in image_hashes:
            os.remove(image_path)
            break
        else:
            # Otherwise, add the hash to the dictionary
            image_hashes[image_hash] = image_path
    return image_hashes


def delete_duplicate_images(folder):
    # Create a dictionary to store image hashes
    image_hashes = {}
    # Calculate the hash for each image in the folder
    for filename in q(os.listdir(folder), desc=f"Deleting duplicates..."):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder, filename)
            image_hashes = _check_duplicate(image_path, image_hashes)


def split_and_overlap_images(img_dir, output_dir, number_of_imgs, stride_ratio=0.5):
    print(
        f"Running split_and_overlap_images with:\nimg_dir: {img_dir}\noutput_dir: {output_dir}\nF: {number_of_imgs}\nstride_ratio: {stride_ratio}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Determine the horizontal and vertical split count based on F (assuming F is a square number)
    split_count = int(math.sqrt(number_of_imgs))
    print(f"Each image will be split into {split_count}x{split_count} segments.")

    # Iterate over the images in the dataset
    for file_name in tqdm(os.listdir(img_dir)):
        if file_name.endswith('.jpg'):
            # Read the image
            img = cv2.imread(os.path.join(img_dir, file_name))

            # Calculate the width and height of the split images
            height, width, _ = img.shape
            new_width = width // split_count
            new_height = height // split_count

            # Calculate the stride size based on the stride_ratio
            stride_size = int(new_width * (1 - stride_ratio))

            # Split the image
            tile_counter = 0
            for i in range(0, height - new_height + 1, stride_size):
                for j in range(0, width - new_width + 1, stride_size):
                    # Extract the image segment
                    new_image = img[i:i + new_height, j:j + new_width]

                    # Save the new images
                    new_file_name = file_name.replace('.jpg', f'_{tile_counter}{random.randint(1,100)}.jpg')
                    cv2.imwrite(os.path.join(output_dir, new_file_name), new_image)
                    tile_counter += 1


# Specify the folder containing the images
img_dir = r'C:\Users\hadjm\Downloads\m'
output_dir = r'C:\Users\hadjm\Downloads\m\tiles'

# Specify the number of tiles and the overlap
number_of_imgs = 9
stride_ratio = 0.1

split_and_overlap_images(img_dir, output_dir, number_of_imgs, stride_ratio)
