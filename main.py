
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

X = np.array([-1.0,0.0,1.0,2.0,3.0,4.0])
Y = np.array([-3.0,-1.0,1.0, 3.0, 5.0,7.0])

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(X,Y, epochs=120)

newX = 5.0
print(f"when my X values {newX}, the Y value will be {model.predict([newX])}")
print()
