import numpy as np
import tensorflow as tf
import os
from keras.models import load_model
from keras.src.legacy.preprocessing.image import ImageDataGenerator  # Image data preprocessing
import matplotlib.pyplot as plt  # Matplotlib for plotting
import seaborn as sns  # Seaborn for visualizing confusion matrix

# GPU configuration (if applicable)
gpus = tf.config.list_physical_devices('GPU')  # List available GPUs
if gpus: 
    try:
        # Memory allocation for GPU
        for gpu in gpus:
            tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=5292)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')  # List logical GPUs
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Error handling for GPU configuration
        print(e)


# Define the data directories
data_dir = 'data'  # Directory containing data
test_dir = os.path.join(data_dir, 'test')  # Test data directory
test_datagen = ImageDataGenerator(rescale=1./255)  # Test data generator
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle=False)  # Set shuffle to False for consistent label ordering

# Load your trained model
model = load_model('ai_real_image_classifier_resnet101.keras')

save_dir = 'metrics_graphs'

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    return figure

# Predict the values from the test_generator
test_generator.reset()  # Reset the generator to be sure of the order
predictions = model.predict(test_generator, steps=test_generator.n//test_generator.batch_size+1)
y_pred = np.argmax(predictions, axis=1)  # Convert predictions classes to one hot vectors 

# True labels of the test data
y_true = test_generator.classes

# Compute the confusion matrix
cm = tf.math.confusion_matrix(y_true, y_pred)

# Plotting the confusion matrix
plot_confusion_matrix(cm, class_names=test_generator.class_indices.keys())