import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import losses_utils
from tensorflow import keras

# X = np.array([-1.0,0.0,1.0,2.0,3.0,4.0])
# Y = np.array([-3.0,-1.0,1.0, 3.0, 5.0,7.0])
#
# model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# model.compile(optimizer='sgd', loss='mean_squared_error')
#
# model.fit(X,Y, epochs=120)
#
# newX = 5.0
# print(f"when my X values {newX}, the Y value will be {model.predict([newX])}")
# print()

(X_train,Y_train),(X_test, Y_test) = tf.keras.datasets.mnist.load_data()
# print(X_train)
# print(X_test)

#print(X_train[0].shape)
# print(X_train[0])
#
# plt.matshow(X_train[0])
# plt.show()
#
# print(Y_train[0])

X_train = X_train/255
X_test = X_test/255

X_train_flattened = X_train.reshape(len(X_train),28 * 28)
print(X_train_flattened.shape)

X_test_flattened = X_test.reshape(len(X_test),28 * 28)
print(X_test_flattened.shape)

# myloss = tf.keras.losses.SparseCategoricalCrossentropy(
#     from_logits=False,
#     ignore_class=None,
#     reduction=losses_utils.ReductionV2.AUTO,
#     name='sparse_categorical_crossentropy'
# )

myloss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model = tf.keras.Sequential([tf.keras.layers.Dense(units=10, input_shape=(784,), activation='relu')])
model.compile(optimizer= "sgd", loss= myloss, metrics='accuracy')
model.fit(X_train_flattened, Y_train, epochs=5)

evaluate = model.evaluate(X_test_flattened, Y_test)
print(evaluate)

plt.matshow(X_test[5])
plt.show()

predict_value = model.predict(X_test_flattened)
res = predict_value[5]
print(np.argmax(res))