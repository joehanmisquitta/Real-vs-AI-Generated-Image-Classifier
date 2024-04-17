import os
from keras.src.legacy.preprocessing.image import ImageDataGenerator  # Image data preprocessing
from keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm  # Import tqdm for progress bar

# Define directories
data_dir = 'data'  # Main directory containing subfolders
input_dir = os.path.join(data_dir, 'train')  # Directory containing subfolders with class names
output_dir = os.path.join(data_dir, 'augmented_train')  # Directory to save augmented images

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Flow from directory and augment images
for root, dirs, files in os.walk(input_dir):
    for subdir in dirs:
        class_input_dir = os.path.join(input_dir, subdir)
        class_output_dir = os.path.join(output_dir, subdir)
        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)

        for filename in tqdm(os.listdir(class_input_dir), desc=f"Augmenting images in {subdir}"):
            img_path = os.path.join(class_input_dir, filename)
            img = load_img(img_path, target_size=(224, 224))  # Resized images to match the expected input size
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)  # Reshape to (1, height, width, channels)

            # Generate batches of augmented images and save to output directory
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=class_output_dir, save_prefix='aug', save_format='jpeg'):
                i += 1
                if i >= 10:  # Generate 10 augmented images per input image
                    break  # Break loop to avoid infinite generation