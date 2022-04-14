import string
from flask import Flask,request,url_for,render_template,flash,redirect
from imageio import imsave,imread
import os
import base64
from load import *
from tensorflow import keras
from keras.preprocessing.image import load_img
import keras.models
import numpy as np
import cv2
from keras.applications.vgg16 import preprocess_input

global model,graph

model=init()

path=os.path.join('static','Uploads')
os.makedirs(path,exist_ok=True)

img_folder='static/Uploads'
allowed_types={'png','jpg'}

def allowed_file(file_name):
    return '.' in file_name and file_name.rsplit('.')[-1].lower() in allowed_types



app=Flask(__name__)

@app.route('/')
def home():
    return render_template('img_upload.html')

@app.route('/',methods=['POST'])
def upload():
    if 'img_file' not in request.files:
        msg='img_file not in request.files.'
        return render_template('msg.html',massage=msg)
        
    else:
        file=request.files['img_file']
        try:
            if allowed_file(file.filename):
                file.save(os.path.join('static/Uploads/' + file.filename))
                # msg="Image successfully saved in 'static/Uploads'"
                
                return render_template('img_upload.html' ,file_name = file.filename)
            else:
                msg="file format must be '.png' or '.jpg' "
                return render_template('msg.html',massage = msg)

        except Exception as ex:
            
            return redirect(request.url)

@app.route('/display/<filename>')
def display(filename):
    return redirect(url_for('static',filename='Uploads/' + filename))

@app.route('/predict',methods=['POST'])
def predict():
    class_names=['cat','dog','wild']
    
    img=load_img('static/Uploads/flickr_dog_000131.jpg')
    img=np.resize(img,(150,150,3))
 
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)
    feature = np.reshape(feature, (-1, 3))
    pred=feature.argmax()
    print('********',pred)
    print(str(feature))
    msg=class_names[pred]
    return render_template('msg.html',massage=msg)
	
