import os
import shutil

'''
OutLine of the directory before Merger:
Data/
│
├── train/
│   ├── Real/          # Training real images
│   └── Fake/     # Training AI-generated images
│
├── validation/
│   ├── Real/          # Validation real images
│   └── Fake/     # Validation AI-generated images
│
└── test/
    ├── Real/          # Test real images
    └── Fake/     # Test AI-generated images


Datatwo/
│
├── train/
│   ├── Real/          # Training real images
│   └── Fake/     # Training AI-generated images
│
├── validation/
│   ├── Real/          # Validation real images
│   └── Fake/     # Validation AI-generated images
│
└── test/
    ├── Real/          # Test real images
    └── Fake/     # Test AI-generated images

Target Directory Structure:
Data/
│
├── train/
│   ├── Real/          # Training real images
│   └── Fake/     # Training AI-generated images
│
├── validation/
│   ├── Real/          # Validation real images
│   └── Fake/     # Validation AI-generated images
│
└── test/
    ├── Real/          # Test real images
    └── Fake/     # Test AI-generated images
'''


# Define the base directories
source_dir = 'Data2'
target_dir = 'Data'

# Define the categories and subsets
categories = ['Real', 'Fake']
subsets = ['train', 'validation', 'test']

# Function to merge two directories
def merge_directories(source, target):
    for item in os.listdir(source):
        source_item = os.path.join(source, item)
        target_item = os.path.join(target, item)
        if os.path.isfile(source_item):
            # If the file already exists in the target directory, rename it before moving
            if os.path.exists(target_item):
                base, extension = os.path.splitext(item)
                target_item = os.path.join(target, f"{base}_duplicate{extension}")
            shutil.move(source_item, target_item)
        elif os.path.isdir(source_item):
            # If the directory does not exist in the target directory, create it
            if not os.path.exists(target_item):
                os.makedirs(target_item, exist_ok=True)
            merge_directories(source_item, target_item)

# Merge the datasets
for subset in subsets:
    for category in categories:
        source_path = os.path.join(source_dir, subset, category)
        target_path = os.path.join(target_dir, subset, category)
        merge_directories(source_path, target_path)

print("Datasets have been successfully merged.")
