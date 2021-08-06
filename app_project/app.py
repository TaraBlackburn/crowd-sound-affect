from flask import Flask, render_template, request, redirect, flash, request, redirect, url_for, send_from_directory
import speech_recognition as sr
from tensorflow.keras.applications.inception_v3 import preprocess_input
import tensorflow as tf
from tensorflow import keras
import os
from model import Inception_v3model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np

"""
app = Flask(__name__)
UPLOAD_FOLDER = '/home/pteradox/Galvanize/capstones/crowd-sound-affect/app_project/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'wav', 'mp3', 'm4a', 'json'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#function to see if file is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('download_file', name=filename))
    return '''
            <!doctype html>
            <title>Upload new File</title>
            <h1>Upload Audio or Spectrogram file</h1>
            <form method=post enctype=multipart/form-data>
            <input type=file name=file>
            <input type=submit value=Upload>
            </form>
            '''


@app.route('/uploads/<name>')
def download_file(name):
    file = 'uploads'
    final_model = tf.keras.models.load_model('model_checkpoints')
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    class_dict = {0:'Approval', 1:'Disapproval', 2:'Neutral'}
    classed = np.argmax(final_model.predict(img_preprocessed))
    return send_from_directory(class_dict[classed], name)

"""



##############################################################################


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
final_model = tf.keras.models.load_model('./model_checkpoints')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    file = request.form
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    class_dict = {0:'Approval', 1:'Disapproval', 2:'Neutral'}
    classed = np.argmax(final_model.predict(img_preprocessed))

    return render_template('index.html', prediction_text='Spectro gram is {} with {} accuracy'.format(classed, class_dict[classed] ))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = final_model.predict(data)

    output = prediction
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
