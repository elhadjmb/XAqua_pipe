import os
import torch
from PIL import Image
from torchvision import models, transforms
from torch.nn.functional import cosine_similarity
import cv2
import numpy as np
from tqdm import tqdm as q

# Load the pre-trained ResNet model
model = models.resnet152(pretrained=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model = model.eval()
print(device)

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_image_features(image_path):
    # Load and transform the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    # Move the image tensor to the same device as the model
    image = image.to(device)

    # Use the model to extract features
    with torch.no_grad():
        features = model(image)

    return features


def delete_duplicate_images(folder, similarity_threshold=0.97):
    # Create a dictionary to store image features
    image_features = {}
    counter = 0
    # Calculate the features for each image in the folder
    for filename in q(os.listdir(folder), desc='Duplicate images deletion...'):
        if filename.endswith('tile.jpg') or filename.endswith('tile.png'):
            image_path = os.path.join(folder, filename)
            try:
                features = get_image_features(image_path)
            except:
                continue
            # Compare the features to those of the other images
            for other_features in image_features.values():
                similarity = cosine_similarity(features, other_features)

                # If the similarity is above the threshold, delete the image
                if similarity.item() > similarity_threshold:
                    os.remove(image_path)
                    counter += 1
                    break
            else:
                # Otherwise, add the features to the dictionary
                image_features[filename] = features

    print("Deleted ", counter)


def calculate_focus_measure(image_path):
    """
    Calculate the focus measure (variance of Laplacian) for an image.

    Parameters:
        image_path (str): The path to the image.

    Returns:
        float: The focus measure.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    return fm


def delete_blurry_images(folder, dynamic_threshold_factor=0.8):
    """
    Delete blurry images based on a dynamic threshold.

    Parameters:
        folder (str): The folder containing the images.
        dynamic_threshold_factor (float): Factor to scale the average focus measure.

    Returns:
        None
    """
    focus_measures = []

    # First pass to calculate average focus measure
    for filename in q(os.listdir(folder), desc='Blurry images deletion...'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder, filename)
            fm = calculate_focus_measure(image_path)
            focus_measures.append(fm)

    # Calculate dynamic threshold
    avg_focus_measure = np.mean(focus_measures)
    dynamic_threshold = avg_focus_measure * dynamic_threshold_factor
    i=0
    # Second pass to delete blurry images
    for filename in q(os.listdir(folder)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder, filename)
            fm = calculate_focus_measure(image_path)

            if fm < dynamic_threshold:
                os.remove(image_path)
                i += 1
    print("Deleted",i)



# # Specify the folder containing the images
# folder = r'/home/hadjm/Downloads/Imaegs'
#
# # Delete blurry images in the folder
# delete_blurry_images(folder)
# # Delete duplicate images in the folder
# delete_duplicate_images(folder)
