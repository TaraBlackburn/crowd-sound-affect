import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow import keras
import os
from model import Inception_v3model
from tensorflow.keras.preprocessing import image
import numpy as np

# I just need these to fit my pre-loaded model, other than that there is no use. 
images = '/home/pteradox/Galvanize/capstones/crowd-sound-affect/dataset/step4_split_spectrograms/dataset_training/all_freq'
testing = '/home/pteradox/Galvanize/capstones/crowd-sound-affect/dataset/step4_split_spectrograms/dataset_test/all_freq'
#Pre-trained and pre-loaded model
model = Inception_v3model(images, testing)
class_dict = {0:'Approval', 1:'Disapproval', 2:'Neutral'}

st.write("""
         # Spectrogram Classification
         """
         )
         
st.write("Predict whether a spectrogram converted from an audiofile is going to be approval, disapproval or neural")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image)
    prediction = model.predict_model(image)
    
    if np.argmax(prediction) == 0:
        st.write("Approval")
    elif np.argmax(prediction) == 1:
        st.write("Disapproval")
    else:
        st.write("Neural")
    a, b, c = prediction
    st.text(f"Probability {a} : Approval, {b}, Disapproval, {c} Neutral")
    st.write(prediction)