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

def beam_search_predictions(model, features, tokenizer, max_length, beam_index=10):
    start = [tokenizer.word_index['startseq']]
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            sequence = pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = model.predict([features, sequence], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                
        start_word = temp
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    final_caption = [idx_to_word(i, tokenizer) for i in start_word]
    final_caption = ' '.join([word for word in final_caption if word not in ['startseq', 'endseq']])
    
    return final_caption
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
        caption =beam_search_predictions(caption_model,features,tokenizer, 34)
        st.write("Generated Caption: ", caption)
