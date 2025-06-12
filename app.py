from __future__ import division, print_function
# coding=utf-8
import sys
import os
import os
from PIL import Image
import glob
import re
import numpy as np
import cv2
# Keras
#import packages and classes
import pandas as pd
import numpy as np
import cv2
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import  MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import pickle
import os
import matplotlib.pyplot as plt
import keras
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Lambda, Activation, Flatten, Input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from Attention import attention #===============importing attention layer
from sklearn.metrics import average_precision_score
import seaborn as sns
from numpy import dot
from numpy.linalg import norm

# Flask utils
from flask import Flask, redirect, url_for, request, render_template

import sqlite3
import pandas as pd
import numpy as np
import pickle
import sqlite3
import random

import smtplib 
from email.message import EmailMessage
from datetime import datetime



app = Flask(__name__)



UPLOAD_FOLDER = 'static/uploads/'

# allow files of a specific type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# function to check the file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#function to normalize bounding boxes
def convert_bb(img, width, height, xmin, ymin, xmax, ymax):
    bb = []
    conv_x = (64. / width)
    conv_y = (64. / height)
    height = ymax * conv_y
    width = xmax * conv_x
    x = max(xmin * conv_x, 0)
    y = max(ymin * conv_y, 0)     
    x = x / 64
    y = y / 64
    width = width/64
    height = height/64
    return x, y, width, height

#define global variables to store processed images, labels and bounding boxes
boundings = []
X = []
Y = []

#define extended kalman filter object
kalman = cv2.legacy.TrackerMOSSE_create()


#define CVC-09 pedestrian dataset path
path = 'Dataset/Annotations'
if os.path.exists('model/X.txt.npy'):
    X = np.load('model/X.txt.npy')#load all processed images
    Y = np.load('model/Y.txt.npy')                    
    boundings = np.load('model/bb.txt.npy')#load bounding boxes
else:
    for root, dirs, directory in os.walk(path):#if not processed images then loop all annotation files with bounidng boxes
        for j in range(len(directory)):
            file = open('Dataset/Annotations/'+directory[j], 'r')
            name = directory[j]
            name = name.replace("txt","png")
            if os.path.exists("Dataset/FramesPos/"+name):
                img = cv2.imread("Dataset/FramesPos/"+name)
                height, width, channel = img.shape
                img = cv2.resize(img, (64, 64))#Resize image
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                lines = file.readlines()
                boxes = []
                for m in range(0,12):
                    boxes.append(0)
                start = 0    
                for i in range(len(lines)):#loop and read all bounding boxes from image
                    if start < 12:
                        line = lines[i].split(" ")
                        x1 = float(line[0])
                        y1 = float(line[1]) 
                        x2 = float(line[2])
                        y2 = float(line[3])
                        xx1 = x1 - x2 / 2
                        yy1 = y1 - y2 / 2
                        x2 = x1 + x2 / 2
                        y2 = y1 + y2 / 2
                        x1 = xx1
                        y1 = yy1
                        x1, y1, x2, y2 = convert_bb(img, width, height, x1, y1, x2, y2)#normalized bounding boxes
                        bbox = (x1 * 64, y1 * 64, x2 * 64, y2 * 64)
                        kalman.init(img, bbox)#apply kalman filter images to correct bounding box lovcations
                        kalman.update(img)
                        boxes[start] = x1
                        start += 1
                        boxes[start] = y1 
                        start += 1
                        boxes[start] = x2
                        start += 1
                        boxes[start] = y2
                        start += 1
                boundings.append(boxes)
                X.append(img)
                Y.append(0)
    X = np.asarray(X)#convert array to numpy format
    Y = np.asarray(Y)
    boundings = np.asarray(boundings)
    np.save('model/X.txt',X)#save all processed images
    np.save('model/Y.txt',Y)                    
    np.save('model/bb.txt',boundings)
#print("Dataset Images Loaded")
#print("Total Images Found in Dataset : "+str(X.shape[0]))

#preprocess images by applying shuffling and then split dataset into train and test
indices = np.arange(X.shape[0])
np.random.shuffle(indices)#shuffle image pixels
X = X[indices]
Y = Y[indices]
boundings = boundings[indices]
#split dataset into train and test where 20% dataset size for testing and 80% for testing
split = train_test_split(X, Y, boundings, test_size=0.20, random_state=42)
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]

#train propose improved YoloV5 with squeeze attention model
#define input shape
input_img = Input(shape=(64, 64, 3))
#create YoloV4 layers with 32, 64 and 512 neurons or data filteration size
x = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(input_img)
x = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(x)
x = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(x)
x = MaxPooling2D((2, 2))(x)
x = attention(return_sequences=True,name='attention')(x)#====#adding squeeze attention model to make improved Yolov5
x = Flatten()(x)
#define output layer with 4 bounding box coordinate and 1 weapan class
x = Dense(256, activation = 'relu')(x)
x = Dense(256, activation = 'relu')(x)
x_bb = Dense(12, name='bb',activation='sigmoid')(x)
x_class = Dense(1, activation='sigmoid', name='class')(x)
#create yolo Model with above input details
yolo_model = Model([input_img], [x_bb, x_class])
#compile the model
yolo_model.compile(Adam(lr=0.0001), loss=['mse', 'binary_crossentropy'], metrics=['accuracy'])
#if os.path.exists("model/yolo_weights.hdf5") == False:#if model not trained then train the model
    #model_check_point = ModelCheckpoint(filepath='model/yolo_weights.hdf5', verbose = 1, save_best_only = True)
hist = yolo_model.fit(trainImages, [trainBBoxes, trainLabels], batch_size=32, epochs=10, validation_data=(testImages, [testBBoxes, testLabels]))
    #f = open('model/yolo_history.pckl', 'wb')
    #pickle.dump(hist.history, f)
    #f.close()    
#else:#if model already trained then load it
    #yolo_model = load_model("model/yolo_weights.hdf5", custom_objects={'attention': attention})
predict = yolo_model.predict(testImages)#perform prediction on test data
    
   
@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')


@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')



@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/predict2',methods=['GET','POST'])
def predict2():
    if request.method == "POST":
         
        print("Entered")
        
        print("Entered here")
        file = request.files['file'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)


            
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        img = cv2.imread(file_path)#read test image
        img = cv2.resize(img, (64, 64))#Resize image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img1 = img.reshape(1,64,64,3)
        predict_value = yolo_model.predict(img1)#perform prediction on test data using extension model
        predict = predict_value[0]#get bounding boxes
        predict = predict[0]
        predicted_label = predict_value[1][0][0] #get predicted label
        flag = True
        start = 0
        while flag:#now loop and plot all detected pedestrains
            if start < 12:
                x1 = predict[start] * 64
                start += 1
                y1 = predict[start] * 64
                start += 1
                x2 = predict[start] * 64
                start += 1
                y2 = predict[start] * 64
                start += 1
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 20:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 1, 1)
                    cv2.putText(img, str(int(predicted_label)+1), (int(x1), int(y1+40)),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)   
                    img_base64 = Image.fromarray(img)
                    img_base64 = img_base64.resize((500, 500))
                    img_base64.save("static/image0.jpg", format="JPEG")
                    return redirect("static/image0.jpg")

            else:
                flag = False
                return render_template('index.html')
        

        

        
              
    return render_template('index.html')



@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(1000,5000)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "evotingotp4@gmail.com"
    msg['To'] = email
    
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("evotingotp4@gmail.com", "xowpojqyiygprhgr")
    s.send_message(msg)
    s.quit()
    return render_template("val.html")

@app.route('/predict1', methods=['POST'])
def predict1():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signup.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signin.html")

@app.route("/notebook")
def notebook1():
    return render_template("Pedestriansdetection.html")


   
if __name__ == '__main__':
    app.run(debug=False)