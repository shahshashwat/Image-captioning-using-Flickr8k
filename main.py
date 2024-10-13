import streamlit as st
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
import pickle
import tensorflow as tf
# Load the Inception model once (outside of the predict function)
feature_extractor = InceptionV3(weights='imagenet', include_top=False, pooling='avg')  # Use pooling to get features directly
caption_model =  tf.keras.models.load_model('image_caption.keras')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

def extract_features(image):
    image = preprocess_input(image)
    features = feature_extractor.predict(image)
    return features

def idx_to_word(integer,tokenizer):
    
    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

def predict_caption(model, features, tokenizer, max_length):
    in_text = "startseq"  # Start token for caption generation

    for i in range(max_length):  # Ensure the loop does not exceed max_length
        sequence = tokenizer.texts_to_sequences([in_text])[0]  # Convert text to sequence
        sequence = pad_sequences([sequence], maxlen=max_length)  # Ensure this uses the correct max_length (34)

        # Debugging shapes (optional to help diagnose issues)
        print("Features shape:", features.shape)
        print("Sequence shape:", sequence.shape)

        # Predict the next word using the model
        y_pred = model.predict([features, sequence])  # Ensure shapes match the model's expected input
        y_pred = np.argmax(y_pred)  # Get the index of the predicted word
        
        word = idx_to_word(y_pred, tokenizer)  # Convert index to word
        
        if word is None:  # Break if no word found
            break
        
        in_text += " " + word  # Append the word to the input text
        
        if word == 'endseq':  # Stop if end token is predicted
            break
            
    return in_text.replace("startseq",'').replace("endseq",'').strip()
# Streamlit app layout
st.title("Image Caption Generator")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the uploaded image
    img = load_img(uploaded_file, target_size=(299, 299))  # InceptionV3 requires 299x299 input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to fit the model input shape

    # Extract features for the uploaded image
    features = extract_features(img_array)

    # Display the image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    # Generate and display the caption (assuming you have the caption prediction function defined)
    if st.button("Generate Caption"):
        caption = predict_caption(caption_model,features,tokenizer, 34)
        st.write("Generated Caption: ", caption)
