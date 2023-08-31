import os
import torch
from PIL import Image
from torchvision import models, transforms
from torch.nn.functional import cosine_similarity

from tqdm import tqdm as q
# Load the pre-trained ResNet model
model = models.resnet50(pretrained=True)
model = model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_features(image_path):
    # Load and transform the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    # Use the model to extract features
    with torch.no_grad():
        features = model(image)

    return features

def delete_duplicate_images(folder, similarity_threshold=0.98):
    # Create a dictionary to store image features
    image_features = {}

    # Calculate the features for each image in the folder
    for filename in q(os.listdir(folder)):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder, filename)
            features = get_image_features(image_path)

            # Compare the features to those of the other images
            for other_features in image_features.values():
                similarity = cosine_similarity(features, other_features)

                # If the similarity is above the threshold, delete the image
                if similarity.item() > similarity_threshold:
                    os.remove(image_path)
                    break
            else:
                # Otherwise, add the features to the dictionary
                image_features[filename] = features

# Specify the folder containing the images
folder = r'C:\Users\hadjm\Downloads\m\tiles'

# Delete duplicate images in the folder
delete_duplicate_images(folder)
