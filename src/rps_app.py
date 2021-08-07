import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
import os
from model import Inception_v3model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
import seaborn as sns 
import shutil
import matplotlib.pyplot as plt

model = load_model('/home/pteradox/Galvanize/capstones/crowd-sound-affect/src/model_checkpoint/my_h5_model_dropouts')

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
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(224, 224),    interpolation=cv2.INTER_CUBIC))/255.
        
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
    a, b, c = prediction[0]
    st.text(f"Probability of Approval: {a}, Disapproval: {b}, Neutral: {c} ")
    st.write(prediction)