# Advanced AI Real Image Classifier using ResNet101-based Deep Learning

Short description of the project.
This Project is still a work in progress

## Overview

Brief overview of the project, including its purpose and main features.

## Installation

Instructions on how to install and set up the project locally. Include any dependencies that need to be installed.

## Usage

Guidance on how to use the project's code. Provide examples or usage scenarios if applicable.

## File Structure

Description of the project's files and directories. Important files are Highlighted.

MoveAugmentedImages.py

**App.py**

DataAugmentation.py

DataDeleter.py

DataMerger.py

DataSplitD1.py

DataSplitD2.py

DataUnZipper.py

DataZIpper.py

**ModelTraining.py**

## Dataset Overview

#### Data Sources:

**CIFAKE: Real and AI-Generated Synthetic Images:** [Dataset 1](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

- Citations for Dataset 1:

  - Bird, J.J. and Lotfi, A., 2024. CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images. IEEE Access.

  - Real images are from Krizhevsky & Hinton (2009), fake images are from Bird & Lotfi (2024). The Bird & Lotfi study is available [here](https://ieeexplore.ieee.org/abstract/document/10409290).

**Fake or Real Competition Dataset:** [Dataset 2](https://www.kaggle.com/datasets/kidonpark1023/fake-or-real-dataset/data)

The dataset used for this project after Data Pre-Processing and Data Augmentation is organized as follows:

- **Train**: Contains 114,000 images divided into 2 classes.
- **Validation**: Consists of 23,000 images divided into 2 classes.
- **Test**: Comprises 23,000 images divided into 2 classes.

Each class represents a different category or label in the dataset.

## Data Structure

The project directory contains the following subfolders:

- **data**: This folder contains the datasets used for training, validation, and testing.

  - **train**: Contains training data.
    - **Fake**: Subfolder containing fake images for training.
    - **Real**: Subfolder containing real images for training.

  - **validation**: Contains validation data.
    - **Fake**: Subfolder containing fake images for validation.
    - **Real**: Subfolder containing real images for validation.

  - **test**: Contains test data.
    - **Fake**: Subfolder containing fake images for testing.
    - **Real**: Subfolder containing real images for testing.


## Model Training

This section describes the process of training the model for the image classification task.

### Data Preparation

The data preparation process involves organizing the dataset for training, validation, and testing, along with applying data augmentation techniques to enhance the diversity of the training data. Here's how the data is prepared:

#### 1.Splitting the First Dataset

Original Data Structure:

- **Data:**
  - `RealArt/`: Training real images
  - `GeneratedArt/`: Training AI-generated images

The first dataset is initially organized into two subfolders: `RealArt/` containing real images and `GeneratedArt/` containing AI-generated images. The script "DataSplitD1.py" splits this dataset into training, validation, and test sets, resulting in the following directory structure:

- **Training Data:**
  - `RealArt/`: Training real images
  - `GeneratedArt/`: Training AI-generated images
  
- **Validation Data:**
  - `RealArt/`: Validation real images
  - `GeneratedArt/`: Validation AI-generated images
  
- **Test Data:**
  - `RealArt/`: Test real images
  - `GeneratedArt/`: Test AI-generated images

This structured dataset is then utilized for training and evaluating the deep learning model.

#### 2. Splitting the Second Dataset

Original Data Structure:

- **Training Data:**
  - `RealArt/`: Training real images
  - `GeneratedArt/`: Training AI-generated images

- **Test Data:**
  - `RealArt/`: Test real images
  - `GeneratedArt/`: Test AI-generated images

The second dataset is initially organized into two subfolders: `Train/` and `Test/` containing two classes of images: `RealArt/` containing real images and `GeneratedArt/` containing AI-generated images. The script "DataSplitD2.py" splits this dataset into training, validation, and test sets, resulting in the following directory structure:

- **Training Data:**
  - `RealArt/`: Training real images
  - `GeneratedArt/`: Training AI-generated images
  
- **Validation Data:**
  - `RealArt/`: Validation real images
  - `GeneratedArt/`: Validation AI-generated images
  
- **Test Data:**
  - `RealArt/`: Test real images
  - `GeneratedArt/`: Test AI-generated images

This structured dataset is then utilized for training and evaluating the deep learning model.

### Directory Merging

For Code Refer to "DataMerger.py"

This script is designed to merge the contents of two directory structures into a single target directory structure. Here's what it does:

1. **Initial Directory Structure**:
   - The script outlines the directory structure of two source directories (`Data` and `Data2`). Each directory contains subdirectories for different subsets (`train`, `validation`, and `test`), and within each subset, there are subdirectories for different categories (`Real` and `Fake`). This represents a typical setup for organizing image datasets, where `Real` refers to real images and `Fake` refers to AI-generated images.

2. **Target Directory Structure**:
   - The script specifies the target directory structure, which is identical to the structure of the source directories (`Data`). This structure is maintained after merging the contents of the two source directories.

3. **Merging Process**:
   - The script defines a function `merge_directories` to recursively merge the contents of two directories.
   - For each item in the source directory, whether it's a file or a subdirectory:
     - If it's a file:
       - If a file with the same name already exists in the target directory, the script renames the file by appending "_duplicate" to its name before moving it.
       - The file is then moved from the source directory to the corresponding location in the target directory.
     - If it's a subdirectory:
       - If the subdirectory doesn't exist in the target directory, it is created.
       - The function recursively merges the contents of the subdirectory.

4. **Merge Execution**:
   - The script iterates through each subset (`train`, `validation`, `test`) and each category (`Real`, `Fake`) in both source directories.
   - For each combination of subset and category, it determines the source and target paths and initiates the merging process using the `merge_directories` function.

5. **Completion Message**:
   - After the merge process is complete, the script prints a success message indicating that the datasets have been successfully merged.

This script facilitates the consolidation of two datasets into a single Dataset.

### Data Augmentation

This section describes how to perform data augmentation on a set of images using Keras and its `ImageDataGenerator`.

For the Code Refer to "DataAugmentation.py"

#### Steps to Augment Data

1. **Input Directory Setup**:
    - Ensure you have a main directory (`data_dir`) containing subfolders (`input_dir`) with images categorized into different classes.

2. **Output Directory Setup**:
    - An output directory (`output_dir`) is created to store the augmented images. Each class will have its own subfolder within this directory.

3. **Data Augmentation Parameters**:
    - Various augmentation parameters are defined using the `ImageDataGenerator` object:
        - Rotation Range: Images can be rotated by a certain degree specified in `rotation_range`. Default: 20.
        - Width Shift Range: Shifting the width of the image within a specified range (`width_shift_range`). Default: 0.2.
        - Height Shift Range: Shifting the height of the image within a specified range (`height_shift_range`). Default: 0.2.
        - Shear Range: Applying shear transformations to the image (`shear_range`). Default: 0.2.
        - Zoom Range: Zooming into the image by a certain factor (`zoom_range`). Default: 0.2.
        - Horizontal Flip: Flipping the image horizontally (`horizontal_flip`). Default: True.
        - Fill Mode: Strategy used for filling in newly created pixels, typically used when there's transformation resulting in empty areas (`fill_mode`). Default: 'nearest'.

4. **Flow and Augmentation**:
    - The script iterates through each class folder in the input directory.
    - For each image in a class folder:
        - The image is loaded and resized to a standard size (`target_size`).
        - The image is converted to an array and reshaped to match the input requirements of Keras (`img_to_array`).
        - Augmentation is performed using `ImageDataGenerator.flow` method, generating batches of augmented images.
        - Augmented images are saved to the output directory.

5. **Number of Augmented Images**:
    - By default, the script generates 10 augmented images per input image. This number can be adjusted as needed by modifying the `i >= 10` condition.

6. **Progress Monitoring**:
    - Progress is monitored using the `tqdm` library, providing a progress bar for each class's augmentation process.

7. **Output**:
    - After running the script, the output directory will contain the augmented images, organized into subfolders by class.


### Libraries Used

- TensorFlow: Deep learning library for building and training neural networks.
- NumPy: Library for numerical computations.
- Keras: High-level neural networks API, running on top of TensorFlow.
- Matplotlib: Library for creating visualizations in Python.

### Model Architecture

The model is based on the ResNet101 architecture, which is a deep convolutional neural network known for its effectiveness in image classification tasks. We utilize transfer learning by loading the pre-trained ResNet101 model without the top classification layer.

The architecture is extended by adding additional layers on top of the base ResNet101 model:
- GaussianNoise layer with a standard deviation of 0.1 is added for regularization.
- GlobalAveragePooling2D layer to reduce spatial dimensions and summarize feature maps.
- Dense layer with ReLU activation and L2 regularization.
- Dropout layer with a dropout rate of 0.4 for regularization.
- Output layer with softmax activation for binary classification.

### Training Process

The model is compiled using the Adam optimizer with a custom learning rate schedule. We employ the PolynomialDecay schedule to dynamically adjust the learning rate during training.

The training data is prepared using data augmentation techniques such as rotation, shifting, shearing, zooming, and horizontal flipping. The ImageDataGenerator class from Keras is used for data augmentation.

During training, a custom callback named StopTrainingOnValidationLoss is utilized to monitor the validation loss. Training is stopped if the validation loss exceeds the training loss, and the weights from the epoch with the lowest validation loss are saved.

### Evaluation Metrics

The model is evaluated using various metrics on the test set, including:
- Loss: Binary cross-entropy loss function.
- Accuracy: Classification accuracy.
- Precision: Precision score.
- Recall: Recall score.
- AUC: Area under the ROC curve.
- F1 Score: F1 score for binary classification.

### Results and Visualizations

The training history, including loss and accuracy, is plotted using Matplotlib. Additionally, the model architecture is visualized using the plot_model function from Keras.

The trained model is saved in the Keras HDF5 format for future use.

