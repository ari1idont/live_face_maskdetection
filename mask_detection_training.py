
#Training

import os
import PIL
import cv2


from PIL import Image,ImageDraw
import matplotlib.pyplot as plt

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



#making the datasets
path1= 'C:/Users/arindam bhattacharya/Documents/TSF_Spark_internship_projects/experiements/dest_folder/train/with_mask'
path2= 'C:/Users/arindam bhattacharya/Documents/TSF_Spark_internship_projects/experiements/dest_folder/train/without_mask'
xTrain=[]
yTrain=[]
for f in os.listdir(path1):
    image=cv2.imread(os.path.join(path1, f))
    image=cv2.resize(image,(150,150),interpolation=cv2.INTER_AREA)
    image=np.array(image)
    image=image.astype('float32')
    image/=255
    xTrain.append(image)
ym_l=len(xTrain)
for i in range(ym_l):
    yTrain.append([1,0])
for f in os.listdir(path2):
    #xTrain.append(Image.open(os.path.join(path2, f)))
    image=cv2.imread(os.path.join(path2, f))
    image=cv2.resize(image,(150,150),interpolation=cv2.INTER_AREA)
    image=np.array(image)
    image=image.astype('float32')
    image/=255
    xTrain.append(image)

ynm_l=len(xTrain)-ym_l
for i in range(ynm_l):
    yTrain.append([0,1])

path3= 'C:/Users/arindam bhattacharya/Documents/TSF_Spark_internship_projects/experiements/dest_folder/test/with_mask'
path4= 'C:/Users/arindam bhattacharya/Documents/TSF_Spark_internship_projects/experiements/dest_folder/test/without_mask'
xTest=[]
yTest=[]
for f in os.listdir(path3):
    #xTest.append(Image.open(os.path.join(path3, f)))
    image=cv2.imread(os.path.join(path3, f))
    image=cv2.resize(image,(150,150),interpolation=cv2.INTER_AREA)
    image=np.array(image)
    image=image.astype('float32')
    image/=255
    xTest.append(image)

ym_l_test=len(xTest)

for i in range(ym_l_test):
    yTest.append([1,0])
for f in os.listdir(path4):
    #xTest.append(Image.open(os.path.join(path4, f)))
    image=cv2.imread(os.path.join(path4, f))
    image=cv2.resize(image,(150,150),interpolation=cv2.INTER_AREA)
    image=np.array(image)
    image=image.astype('float32')
    image/=255
    xTest.append(image)

ynm_l_test=len(xTest)-ym_l_test
for i in range(ynm_l_test):
    yTest.append([0,1])

plt.imshow(xTrain[5])
print(yTrain[5])
plt.imshow(xTest[129])
print(yTest[129])
#creating the model

#greyscale
#resize
#model
#createmodel
#fitmodel
#check acc -> should be above 90
#and voila we should be done to submit tommorrow.
xTrain=np.asarray(xTrain).astype('float32')
yTrain=np.array(yTrain)
xTest=np.asarray(xTest).astype('float32')
yTest=np.array(yTest)

def create_model():
    
    input_shape = (150, 150,3)
    model=Sequential()
    model.add(Convolution2D(100,kernel_size=(3,3),activation='relu',input_shape=input_shape))
   
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(100,(3,3),activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    #model.add(Dense(256,activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(num_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
model=create_model()
model.fit(xTrain,yTrain,validation_data=(xTest,yTest),epochs=10,verbose=1)
print('Model was trained')
scores=model.evaluate(xTest,yTest,verbose=0)
print("error: %.2f%%"%(100-scores[1]*100))


model_json=model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
#model.save('model.h5')
print('saved')

# Training accuracy 95.4%




