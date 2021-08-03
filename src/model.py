from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import image
from PIL import Image
import glob


class load_data():
    def __init__(self, file):
        self.file = file
    
    # def files_to_arrays(self):
    #     images_approval = []
    #     for file in glob.iglob(r'../dataset/step4_split_spectrograms/dataset_images_training/approval_all/approval_*/appl*.png'):
    #         images_approval.append(image.imread(file))
    #     return images_approval