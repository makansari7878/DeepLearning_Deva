import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


dataset_train = pd.read_csv( r"C:\Users\Personal\Desktop\New folder\Google_Stock_Price_Train.csv")
#print(dataset_train)
trainig_set = dataset_train.iloc[:,1:2].values
#print(trainig_set)

#scaling
sc = MinMaxScaler(feature_range=(0,1))
trainig_set_scaled = sc.fit_transform(trainig_set)
print(trainig_set_scaled)

X_train = []
Y_train =[]

#pulling a chunk of 60 records and the 61 record will be the o/p
for i in range(60,1258):
    X_train.append(trainig_set_scaled[i-60:i, 0])
    Y_train.append(trainig_set_scaled[i,0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

#Reshaping
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

model = Sequential()


# passing only 50 units to next input
# using dropout .. removing the 10 units
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=50, batch_size=32)



















