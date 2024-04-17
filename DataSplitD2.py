import os
import shutil
from sklearn.model_selection import train_test_split

'''
OutLine of the directory before Split:
Datatwo/
├── train/
│   ├── FAKE/     # Training real images
│   └── REAL/     # Training AI-generated images
└── test/
    ├── FAKE/     # Test real images
    └── REAL/     # Test AI-generated images 

Target Directory Structure:
Datatwo/
│
├── train/
│   ├── RealArt/          # Training real images
│   └── GeneratedArt/     # Training AI-generated images
│
├── validation/
│   ├── RealArt/          # Validation real images
│   └── GeneratedArt/     # Validation AI-generated images
│
└── test/
    ├── RealArt/          # Test real images
    └── GeneratedArt/     # Test AI-generated images
'''

# Define the base directory where the 'train' and 'test' folders are located
base_dir = 'Data2'

# Define the paths for the training, validation, and test directories
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Create the validation directory
os.makedirs(validation_dir, exist_ok=True)

# Define the categories
categories = {'FAKE': 'GeneratedArt', 'REAL': 'RealArt'}

# Split the training data into training and validation sets
for category, new_category in categories.items():
    # Create new category directories in both training and validation directories
    os.makedirs(os.path.join(train_dir, new_category), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, new_category), exist_ok=True)
    
    # Get the list of images in the original training category directory
    original_category_dir = os.path.join(train_dir, category)
    images = os.listdir(original_category_dir)
    
    # Split the images into training and validation sets
    train_images, validation_images = train_test_split(images, test_size=0.2, random_state=42)
    
    # Move the images to the new training category directory
    for image in train_images:
        shutil.move(os.path.join(original_category_dir, image), os.path.join(train_dir, new_category))
    
    # Move the images to the new validation category directory
    for image in validation_images:
        shutil.move(os.path.join(original_category_dir, image), os.path.join(validation_dir, new_category))

# Rename the test category directories
for category, new_category in categories.items():
    os.rename(os.path.join(test_dir, category), os.path.join(test_dir, new_category))

print("Data separation complete.")
