from flask import Flask, render_template, request, redirect, flash, request, redirect, url_for, send_from_directory
import speech_recognition as sr
import os
from werkzeug.utils import secure_filename

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
    <title>Upload Audio or Spectrogram file</title>
    <h1>Upload Audio or Spectrogram file</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

    
# @app.route("/", methods=["GET", "POST"])
# def index():
#     transcript = ""
#     if request.method == "POST":
#         print("FORM DATA RECEIVED")

#         if "file" not in request.files:
#             return redirect(request.url)

#         file = request.files["file"]
#         if file.filename == "":
#             return redirect(request.url)

#         if file:
#             recognizer = sr.Recognizer()
#             audioFile = sr.AudioFile(file)
#             with audioFile as source:
#                 data = recognizer.record(source)
#             transcript = recognizer.recognize_google(data, key=None)

#     return render_template('index.html', transcript=transcript)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)