# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 10:38:26 2021

@author: arindam bhattacharya
"""

#Detection code run this code after training

import os
import PIL
import cv2
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
import win32gui
import keras
from keras.models import Sequential
from keras.models import model_from_json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras import backend as k
import cv2
from keras.models import model_from_json








json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)




#face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_clsfr = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

labels_dict={0:'mask',1:'no mask'}
color_dict={0:(0,255,0),1:(0,0,255)}

size = 4
#webcam = cv2.VideoCapture(0) #Use camera 0


# We load the xml file
#classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
window = "OpenCV_window"
webcam = cv2.VideoCapture(0) 
print(webcam, "is open:", webcam.isOpened())
frame_counter=1

while True:
    frame_counter=frame_counter+1
    #(rval, im) = webcam.read()
    check,im=webcam.read()
    if check:
    
        
        im=cv2.flip(im,1,1) #Flip to act as a mirror
    
        # Resize the image to speed up detection
        mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
    
        plt.imshow(mini)
        # detect MultiScale / faces 
        faces = face_clsfr.detectMultiScale(mini)
    
        # Draw rectangles around each face
        for f in faces:
            (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
            #Save just the rectangle faces in SubRecFaces
            face_img = im[y:y+h, x:x+w]
            resized=cv2.resize(face_img,(150,150))
            normalized=resized/255.0
            plt.imshow(normalized)
            reshaped=np.reshape(normalized,(1,150,150,3))
            reshaped = np.vstack([reshaped])
            result=model.predict(reshaped)
            #print(result)
            
            label=np.argmax(result,axis=1)[0]
            print(result)
          
            cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
            cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            
        # Show the imageq
        cv2.imshow('LIVE',   im)
        key = cv2.waitKey(1)
        
    # if Esc key is press then break out of the loop 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()


