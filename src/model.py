import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
import seaborn as sns 
import shutil
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.preprocessing.image import DirectoryIterator


class Scratch_Model():

    def __init__(self, train_path, valid_path, test_path, batch_size=32): 
        """Takes in a path to a directory"""
        self.train_path = train_path
        self.test_path = test_path
        self.valid_path = valid_path
        self.batch_size = batch_size

    def fit(self):
        """From the path will use a DirectoryIterator and fit the spectrograms with ImageDataGenerator"""
        self.images_train = ImageDataGenerator(rescale=1./255,
                                                shear_range=0.2,
                                                zoom_range=0.2,
                                                horizontal_flip=True)\
                .flow_from_directory(directory=self.train_path, target_size=(224, 224), batch_size=self.batch_size)
        self.images_valid = ImageDataGenerator(rescale=1. / 255)\
                .flow_from_directory(directory=self.valid_path, target_size=(224, 224), batch_size=self.batch_size)
        self.images_test = ImageDataGenerator(rescale=1. / 255)\
                .flow_from_directory(directory=self.test_path, target_size=(224, 224), batch_size=self.batch_size, shuffle=False)
        

    def model_build(self, learningrate=.001):
        """Builds a model"""
        self.learningrate = learningrate
        self.model = Sequential([
                    Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding = 'same', input_shape=(224, 224, 3)),
                    AveragePooling2D(pool_size=(2,2), strides=2),
                    Dropout(.1),
                    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
                    MaxPool2D(pool_size=(4,4), strides=3),
                    Dropout(.3),
                    Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'same'),
                    MaxPool2D(pool_size=(2,2), strides=2),
                    Dropout(.3),
                    Flatten(),
                    Dense(units=3, activation='softmax')
                    ])
        self.model.compile(optimizer=Adam(learning_rate=learningrate), loss='categorical_crossentropy', metrics=['accuracy'])
        self.callback = [tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=2), tf.keras.callbacks.ModelCheckpoint('model_weights/model_weights{epoch:02d}', save_freq=5)]

    def model_fit(self, epochs=50):
        """Fits the model"""
        self.epochs = epochs
        self.model.fit(x=self.images_train,
                        steps_per_epoch=len(self.images_train),
                        validation_data=self.images_valid,
                        validation_steps=len(self.images_valid),
                        epochs=self.epochs,
                        verbose=1, callbacks=[self.callback])
        self.model.save("my_new_model_compact2")
        self.model.save('my_h5_model_compact2',save_format='h5')
        self.model.save('my_keras_compact2.keras')

    def conf_matrix(self):
        Y_pred = self.model.predict(self.images_test, len(self.images_test)// self.batch_size+1)
        y_pred = np.argmax(Y_pred, axis=1)
        print('Confusion Matrix')
        cm = confusion_matrix(self.images_test.classes, y_pred)
        print(cm)
        print('Classification Report')
        target_names = ['Approval', 'Disapproval', 'Neutral']
        print(classification_report(self.images_test.classes, y_pred, target_names=target_names))
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['Approval', 'Disapproval', 'Neutral']); 
        ax.yaxis.set_ticklabels(['Approval', 'Disapproval', 'Neutral']);

    def evaluate_model(self):
        """pre-loaded testing data."""
        final_model = tf.keras.models.load_model('/home/pteradox/Galvanize/capstones/crowd-sound-affect/src/model_checkpoint/my_h5_model_strides')
        final_model.evaluate(self.images_test)
    



if '__name__' == '__main__':
    pass