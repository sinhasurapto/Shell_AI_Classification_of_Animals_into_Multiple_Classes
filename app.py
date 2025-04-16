# Import modules
import streamlit as st 
import tensorflow as tf 
import numpy as np 
import gdown
import os

# File information
url = "https://drive.google.com/uc?id=1kWyI-se7PL3L1Y8cagubbgM_VvQi3D5l"
file_id = "1kWyI-se7PL3L1Y8cagubbgM_VvQi3D5l"
model_path = "multiclass_animal_recognizer.keras"

# If path not exists then download
if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)
    st.warning("Model has been downloaded...")

# Class names
class_names = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 
'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 
'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 
'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 
'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 
'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon',
'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 
'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 
'wolf', 'wombat', 'woodpecker', 'zebra']

# Perform prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    input_arr /= 255.0
    prediction = model.predict(input_arr)
    return class_names[np.argmax(predictions)]

# Set up the application
st.sidebar.title('Animal Recognition System')

# To upload an image
test_image = st.file_uploader("Choose an image: ")
if (st.button("Show Image")):
    st.image(test_image, width=4, use_container_width=True)

# Display the prediction on clicking button
if (st.button('Predict')):
    result = model_prediction(test_image)
    st.success(f"Model is predicting it is {result}.")

