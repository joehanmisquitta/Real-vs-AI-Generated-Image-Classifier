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

Describe the structure of the project's files and directories. Highlight important files or directories.

## Model Training

This section describes the process of training the model for the image classification task.

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

### Data Preparation

Explain how the data is prepared for training, including data augmentation techniques used.

### Model Architecture

Provide details about the architecture of the model used for training. Include information about pre-trained models, custom layers, and regularization techniques.

### Training Process

Describe the process of training the model, including the optimization algorithm used, learning rate schedule, and
