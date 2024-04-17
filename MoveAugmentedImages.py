import os
import shutil
from tqdm import tqdm

# Define directories
data_dir = 'data'
augmented_dir = os.path.join(data_dir, 'augmented_train')
train_dir = os.path.join(data_dir, 'train')

# Iterate over classes in the augmented directory
for class_name in os.listdir(augmented_dir):
    class_augmented_dir = os.path.join(augmented_dir, class_name)
    class_train_dir = os.path.join(train_dir, class_name)

    # Ensure the corresponding directory exists in the train directory
    if not os.path.exists(class_train_dir):
        os.makedirs(class_train_dir)

    # Move augmented images to the train directory with progress bar
    for filename in tqdm(os.listdir(class_augmented_dir), desc=f"Moving {class_name}"):
        src = os.path.join(class_augmented_dir, filename)
        dst = os.path.join(class_train_dir, filename)
        shutil.move(src, dst)

print("Augmented images moved to train folder successfully.")
