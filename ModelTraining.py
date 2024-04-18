# Import necessary libraries
import tensorflow as tf  # TensorFlow library for deep learning
import numpy as np  # NumPy library for numerical computations
import os  # OS module for interacting with the operating system
import keras  # Keras library for building deep learning models
from keras.applications.resnet import ResNet101  # Import ResNet50 architecture
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, GaussianNoise  # Different layers for model architecture
from keras.models import Model  # Model class for defining neural network architectures
from keras.regularizers import l1_l2  # Regularization for preventing overfitting
from keras.src.legacy.preprocessing.image import ImageDataGenerator  # Image data preprocessing
from keras.callbacks import EarlyStopping, Callback  # Callbacks for custom actions during training
from keras.initializers import GlorotUniform #initializes the weights using Glorot (Xavier) initialization
#from keras.layers import GaussianNoise  # Adding Gaussian noise to the model
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

# Custom Callback to stop training if validation loss exceeds training loss
class StopTrainingOnValidationLoss(Callback):
    def __init__(self, filepath):
        super(StopTrainingOnValidationLoss, self).__init__()
        self.filepath = filepath
        self.best_weights = None
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        train_loss = logs.get('loss')
        if val_loss is not None and train_loss is not None:
            if val_loss > train_loss:
                print("\nValidation loss is higher than training loss. Stopping training.")
                self.model.stop_training = True
            else:
                if val_loss < self.best_val_loss:
                    print("\nValidation loss is lower than previous best. Saving weights.")
                    self.best_val_loss = val_loss
                    self.best_weights = self.model.get_weights()
                    self.model.save(self.filepath)

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            print("\nLoading weights from the epoch with lowest validation loss.")
            self.model.set_weights(self.best_weights)


# Load the pre-trained ResNet50 model without the top classification layer
base_model = ResNet101(weights=None, include_top=False)  # Changed to ResNet101

# Add new layers on top of the model
x = base_model.output  # Output of the base model
x = GaussianNoise(0.1)(x)  # Add Gaussian noise with a standard deviation of 0.1
x = GlobalAveragePooling2D()(x)  # Global average pooling layer
x = Dense(1024, activation='relu', kernel_initializer=GlorotUniform(), kernel_regularizer=l1_l2(l1=0.02, l2=0.04))(x)  # Dense layer with ReLU activation and L2 regularization
x = Dropout(0.4)(x)  # Dropout layer for regularization
predictions = Dense(2, activation='softmax')(x)  # Output layer with softmax activation

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)  # Combined model

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = True  # Freeze base layers for training

# Define a custom learning rate schedule
train_steps = 7125
lr_schedule = tf.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1e-4,
    decay_steps=train_steps,
    end_learning_rate=1e-5,
    power=2
)

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss='binary_crossentropy',  # Compile model with Adam optimizer and binary crossentropy loss
              metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC(), keras.metrics.F1Score(average=None, threshold=None, name="f1_score", dtype=None)])

# Define the data directories
data_dir = 'data'  # Directory containing data
train_dir = os.path.join(data_dir, 'train')  # Training data directory
validation_dir = os.path.join(data_dir, 'validation')  # Validation data directory
test_dir = os.path.join(data_dir, 'test')  # Test data directory

# Data augmentation for training images
train_datagen = ImageDataGenerator(rescale=1./255)

# Image data augmentation for validation and test sets
validation_datagen = ImageDataGenerator(rescale=1./255)  # Validation data generator
test_datagen = ImageDataGenerator(rescale=1./255)  # Test data generator

# Data generators for training, validation, and test sets
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical')  # Training data generator

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical')  # Validation data generator

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical')  # Test data generator

# Class weights for imbalanced classes
class_weights = {
    0: 1,  # Class 0
    1: len(train_generator.classes) / np.sum(train_generator.classes)  # Class 1
}

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    min_delta=0.001,    # Minimum change to qualify as an improvement
    patience=4,         # Number of epochs with no improvement after which training will be stopped
    verbose=1,          # Verbosity mode
    mode='min',         # 'min' mode because lower validation loss is better
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)

# Define the custom callback
save_path = 'best_model_weights.h5'
stop_on_val_loss = StopTrainingOnValidationLoss(filepath=save_path)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=60,
    callbacks=[early_stopping, stop_on_val_loss],  # List of callbacks
    class_weight=class_weights  # Class weights for imbalanced data
)

# Evaluate the model on the test set
eval_results = model.evaluate(test_generator)
test_loss, test_accuracy, test_precision, test_recall, test_auc, test_f1_score = eval_results

# Print test metrics
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Test precision: {test_precision:.4f}")
print(f"Test recall: {test_recall:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test F1 Score: {test_f1_score[0]:.4f}")  # Extract scalar value for F1 score

# Create a directory to save the graphs if it doesn't exist
save_dir = 'metrics_graphs'
os.makedirs(save_dir, exist_ok=True)

# Define epochs
epochs = range(1, len(history.history['accuracy']) + 1)

# Plot loss
plt.plot(epochs, history.history['loss'], label='Training Loss')
plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(save_dir, 'loss.png'))
plt.close()

# Plot accuracy
plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(save_dir, 'accuracy.png'))
plt.close()

# Plot precision
plt.plot(epochs, history.history['precision'], label='Training Precision')
plt.plot(epochs, history.history['val_precision'], label='Validation Precision')
plt.title('Training and Validation Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()
plt.savefig(os.path.join(save_dir, 'precision.png'))
plt.close()

# Plot recall
plt.plot(epochs, history.history['recall'], label='Training Recall')
plt.plot(epochs, history.history['val_recall'], label='Validation Recall')
plt.title('Training and Validation Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()
plt.savefig(os.path.join(save_dir, 'recall.png'))
plt.close()

# Plot AUC
plt.plot(epochs, history.history['auc'], label='Training AUC')
plt.plot(epochs, history.history['val_auc'], label='Validation AUC')
plt.title('Training and Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.savefig(os.path.join(save_dir, 'auc.png'))
plt.close()

# Plot F1 Score
plt.plot(epochs, history.history['f1_score'], label='Training F1 Score')
plt.plot(epochs, history.history['val_f1_score'], label='Validation F1 Score')
plt.title('Training and Validation F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.savefig(os.path.join(save_dir, 'f1_score.png'))
plt.close()

# Calculate and visualize confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

# Predict classes for the test set
y_pred = model.predict(test_generator).argmax(axis=1)  # Assuming your model outputs probabilities, use argmax to get predicted classes
y_true = test_generator.classes  # True labels of the test data

# Plot confusion matrix
plot_confusion_matrix(y_true, y_pred)

# Save the trained model
print("Saving the model...")
model.save('ai_real_image_classifier_resnet101.keras')  # Save trained model
print("Model saved")

# Plot model architecture
#keras.utils.plot_model(model, "model_architecture.png", show_shapes=True)
