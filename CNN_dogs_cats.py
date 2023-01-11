import os.path
import random
import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import losses_utils

import matplotlib.pyplot as plt
import pickle


DIRECTORY = r"C:\Users\Personal\Desktop\New folder\cat_dogs"
#print(DIRECTORY)
CATEGORIES = ['cats','dogs']
data = []
#Building the image path
for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category )
    label = CATEGORIES.index(category)
    #print(label)

    #print(folder)
    for img in os.listdir(folder):
        img_path = os.path.join(folder,img)
        # print(img_path)
        # break

        # reading the images and converting it to an array
        img_array = cv2.imread(img_path)
        img_array = cv2. resize(img_array, (120,120))
        # plt.imshow(img_array)
        # plt.show()
        # break

        data.append([img_array,label])

#print(len(data))
random.shuffle(data)
print(data[0])

X= []
Y =[]
for features, labels in data:
    X.append(features)
    Y.append(labels)

X = np.array(X)
Y = np.array(Y)

print(X)
print(Y)

# pickle.dump(X, open('X.pkl', 'wb'))
# pickle.dump(Y, open('Y.pkl','wb'))



#creating model

#convert all the pixels into 0,1
X = X/255
Y = Y/255
print(X.shape)

model = Sequential()

#adding multiple convulaiton layers
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

#Flattening

model.add(Flatten())

#23000, 120, 120, 3)  -- im taking 120, 120, 3 from X
model.add(Dense(128, input_shape=X.shape[1:], activation='relu'))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X,Y, epochs=1, validation_split=0.1)
print(model)



