import tensorflow as tf
import streamlit as st
from keras.models import load_model
from keras.preprocessing import image 
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
import numpy as np

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

# Load your trained model
model = load_model('ai_real_image_classifier_resnet101.keras')

# Define a function to preprocess the images to the correct format
def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale the image
    return img_array

# Set up the Streamlit interface
st.title('AI vs Real Image Classifier')
st.write("This app uses a deep learning model to classify images as AI-generated or real.")

# Upload file interface for multiple files
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")
        
        # Preprocess the uploaded image
        preprocessed_image = preprocess_image(uploaded_file)
        
        # Make a prediction
        predictions = model.predict(preprocessed_image)
        confidence_score = np.max(predictions)  # Get the highest probability value as the confidence score

        #For Debugging
        print(predictions)
        print(confidence_score)
        print(predictions[0][1])
        
        # Display the results
        class_names = ['AI-generated', 'Real']
        string_result = class_names[np.argmax(predictions)]
        st.success(f'The image is classified as: {string_result}')
        st.write(f'Confidence Score: {confidence_score:.5f}')  # Display the confidence score rounded to five decimal places