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


model = models.load_model('/home/pteradox/Galvanize/capstones/crowd-sound-affect/src/model_checkpoint/my_h5_model_strides')

class_dict = {0:'Approval', 1:'Disapproval', 2:'Neutral'}

st.write("""
         # Spectrogram Classification
         """
         )
         
st.write("Predict whether a spectrogram converted from an audiofile is going to be approval, disapproval or neural")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(image, dsize=(224, 224)))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("Approval")
    elif np.argmax(prediction) == 1:
        st.write("Disapproval")
    else:
        st.write("Neural")
    # a, b, c = prediction[0]
    st.text(f"Probability {prediction})")
    st.write(prediction)