import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input

# Load your trained model
model = load_model('ai_real_image_classifier_resnet101.keras')

# Define a function to preprocess the images to the correct format
def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = tf.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

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
        
        # Make a prediction using TensorFlow
        predictions = model.predict(preprocessed_image)
        confidence_score = tf.reduce_max(predictions)  # Use TensorFlow's reduce_max
        predicted_index = tf.argmax(predictions, axis=1)  # Use TensorFlow's argmax
        
        # Convert tensors to numpy arrays for displaying
        predictions_array = predictions.numpy()
        confidence_score_array = confidence_score.numpy()
        predicted_index_array = predicted_index.numpy()
        
        # Display the results
        class_names = ['AI-generated', 'Real']
        string_result = class_names[predicted_index_array[0]]
        st.success(f'The image is classified as: {string_result}')
        st.write(f'Confidence Score: {confidence_score_array[0]:.5f}')  # Display the confidence score rounded to five decimal places
