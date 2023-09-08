import math
import random
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import os
from tqdm import tqdm as q

from Xqua.dupdel import delete_blurry_images, delete_duplicate_images

Image.MAX_IMAGE_PIXELS = None  # or a large value like 1000000000


def tile_images(input_dir, output_dir, tile_size=None, num_tiles=None, overlap=(0, 0)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in q(os.listdir(input_dir), desc=f"Tiling {tile_size, num_tiles}"):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(input_dir, filename))
            width, height = img.size
            count = 0

            overlap_x, overlap_y = overlap

            if tile_size:
                tile_width, tile_height = tile_size
            elif num_tiles:
                # Calculate tile dimensions based on the number of tiles
                tile_width = math.ceil(width / math.sqrt(num_tiles))
                tile_height = math.ceil(height / math.sqrt(num_tiles))
            else:
                raise ValueError("Either tile_size or num_tiles must be specified.")

            for i in range(0, height - tile_height + 1, tile_height - overlap_y):
                for j in range(0, width - tile_width + 1, tile_width - overlap_x):
                    left = j
                    upper = i
                    right = j + tile_width
                    lower = i + tile_height
                    tile = img.crop((left, upper, right, lower))

                    tile_filename = os.path.join(output_dir,
                                                 f"{filename.split('.')[0]}_tile_{count}_id_{random.randint(0, 1_000)}__{num_tiles}_{tile_size}_tile.{'jpg' if filename.endswith('.jpg') else 'png'}")
                    tile.save(tile_filename)
                    count += 1


def process_tile(img, j, i, tile_width, tile_height, output_dir, filename, count, num_tiles, tile_size):
    left = j
    upper = i
    right = j + tile_width
    lower = i + tile_height
    tile = img.crop((left, upper, right, lower))

    tile_filename = os.path.join(output_dir,
                                 f"{filename.split('.')[0]}_tile_{count}_id_{random.randint(0, 1_000)}__{num_tiles}_{tile_size}_tile.{'jpg' if filename.endswith('.jpg') else 'png'}")
    tile.save(tile_filename)


def thr_tile_images(input_dir, output_dir, tile_size=None, num_tiles=None, overlap=(0, 0)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with ThreadPoolExecutor(max_workers=12) as executor:
        for filename in q(os.listdir(input_dir), desc=f"Tiling {tile_size, num_tiles}"):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img = Image.open(os.path.join(input_dir, filename))
                width, height = img.size
                count = 0

                overlap_x, overlap_y = overlap

                if tile_size:
                    tile_width, tile_height = tile_size
                elif num_tiles:
                    tile_width = math.ceil(width / math.sqrt(num_tiles))
                    tile_height = math.ceil(height / math.sqrt(num_tiles))
                else:
                    raise ValueError("Either tile_size or num_tiles must be specified.")

                for i in range(0, height - tile_height + 1, tile_height - overlap_y):
                    for j in range(0, width - tile_width + 1, tile_width - overlap_x):
                        executor.submit(process_tile, img, j, i, tile_width, tile_height, output_dir, filename, count,
                                        num_tiles, tile_size)
                        count += 1


ofolder = r'/home/hadjm/Downloads/imgs_t'
ifolder = r'/home/hadjm/Downloads/imgs'

# tile_images(ifolder, ofolder, num_tiles=4, overlap=(50, 50))
# tile_images(ifolder, ofolder, num_tiles=2, overlap=(50, 50))
# tile_images(ifolder, ofolder, num_tiles=16, overlap=(50, 50))
# # delete_blurry_images(ofolder)
# tile_images(ifolder, ofolder, num_tiles=9, overlap=(50, 50))
# # delete_blurry_images(ofolder)
# tile_images(ifolder, ofolder, num_tiles=25, overlap=(50, 50))
# # delete_blurry_images(ofolder)
# # delete_duplicate_images(ofolder)
# # tile_images(ifolder, ofolder, tile_size=(1024, 1024), overlap=(50, 50))
# # # delete_blurry_images(ofolder)
# # delete_duplicate_images(ofolder)
# # tile_images(ifolder, ofolder, tile_size=(256, 256), overlap=(50, 50))
# # delete_blurry_images(ofolder, 0.8)
# delete_duplicate_images(ofolder, 0.95)

tile_images("./dataset/imgs/b_0","./dataset/imgs/b_0/sec_b_0",num_tiles=16, overlap=(0,0))
delete_duplicate_images("./dataset/imgs/b_0/sec_b_0", 0.99)
delete_duplicate_images("./dataset/imgs/b_0/sec_b_0", 0.98)
delete_blurry_images("./dataset/imgs/b_0/sec_b_0", 0.5)
