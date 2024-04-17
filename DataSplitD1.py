import os
import shutil
from sklearn.model_selection import train_test_split

'''
OutLine of the directory before Split:
Data/
├── RealArt/          # Training real images
|── GeneratedArt/     # Training AI-generated images 

Target Directory Structure:
Data/
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


# Define paths
base_dir = 'data'
classes = ['real_images', 'fake_images']
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Create train, validation, and test directories
for dir in [train_dir, val_dir, test_dir]:
    for cls in classes:
        os.makedirs(os.path.join(dir, cls), exist_ok=True)

# Function to split data
def split_data(source, train_dir, val_dir, test_dir, train_size=0.7, val_size=0.15, test_size=0.15):
    files = os.listdir(source)
    train_files, test_files = train_test_split(files, train_size=train_size, test_size=(val_size + test_size))
    val_files, test_files = train_test_split(test_files, train_size=val_size / (val_size + test_size), test_size=test_size / (val_size + test_size))
    
    for file in train_files:
        shutil.copy(os.path.join(source, file), os.path.join(train_dir, file))
    for file in val_files:
        shutil.copy(os.path.join(source, file), os.path.join(val_dir, file))
    for file in test_files:
        shutil.copy(os.path.join(source, file), os.path.join(test_dir, file))

# Split data for each class
for cls in classes:
    print(f"Splitting data for {cls}...")
    split_data(os.path.join(base_dir, cls), os.path.join(train_dir, cls), os.path.join(val_dir, cls), os.path.join(test_dir, cls))

print("Data split complete.")
