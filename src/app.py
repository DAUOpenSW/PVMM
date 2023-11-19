# 다시한번 웹으로 연동해보자

from flask import Flask, request
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify,redirect, url_for, render_template
import os
from Web_V1 import *
app=Flask(__name__)

@app.route("/")
def root():
    return redirect(url_for('home'))

@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/upload', methods=['POST'])
def upload_file():
     weights_paths = ['src/models/weights6.h5']
     curse = CurseDetector(weights_paths)

     file=request.files['audio']
     filename=secure_filename(file.filename)
     file.save(os.path.join('.', filename))
     
     # Now you can use the saved audio file with your recognize_speech() function.
     recognize_speech(filename,curse)
     
     return 'File uploaded and processed successfully'

if __name__ == '__main__':
      app.run(debug=True)
