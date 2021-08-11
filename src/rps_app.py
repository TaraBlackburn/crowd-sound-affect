import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras import models
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import sklearn


# loaded_model = models.load_model('/home/pteradox/Galvanize/capstones/crowd-sound-affect/src/model_checkpoint/my_h5_model_compact')

class_dict = {0:'Approval', 1:'Disapproval', 2:'Neutral'}

st.write("""
         # Spectrogram Classification
         """
         )
         
st.write("Predict whether a spectrogram converted from an audiofile is going to be approval, disapproval or neural")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

def import_and_predict(image_data):
        model = models.load_model('/home/pteradox/Galvanize/capstones/crowd-sound-affect/src/model_checkpoint/my_h5_model_compact')
        image = tf.keras.preprocessing.image.img_to_array(image_data)/255
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        predict = model.predict(image)
        return predict

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image)
    feed = Image.open(file).convert('RGB')
    feed = feed.resize((224,224))
    prediction = import_and_predict(feed)
    
    if np.argmax(prediction) == 0:
        st.write("Approval")
    elif np.argmax(prediction) == 1:
        st.write("Disapproval")
    else:
        st.write("Neural")
    # a, b, c = prediction[0]
    st.text(f"Probability {prediction})")
    st.write(prediction)