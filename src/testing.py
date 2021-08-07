import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
import itertools
from PIL import Image, ImageOps
import cv2
import os, glob
import shutil
import random
import joblib
import pickle
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
from model import Inception_v3model, VGG16model
warnings.simplefilter(action='ignore', category=FutureWarning)


file = '/home/pteradox/Galvanize/capstones/crowd-sound-affect/dataset/step4_split_spectrograms/dataset_test/log/approval/appl000000000014.png'

def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(224, 224),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction


model = load_model('/home/pteradox/Galvanize/capstones/crowd-sound-affect/src/model_checkpoint/my_h5_model')
image = Image.open(file)
prediction = import_and_predict(image, model)
print(prediction)