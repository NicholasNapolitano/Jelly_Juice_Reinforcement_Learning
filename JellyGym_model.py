import numpy as np
import pandas as pd
import csv
import pickle
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import AvgPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from collections import deque

replay_memory = []
learning_rate = 0.001
epsilon = 1.0

model = Sequential()
model.add(Dense(100, use_bias = False, input_dim = 82))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(200, use_bias = False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(324, use_bias = False))
model.add(BatchNormalization())
model.add(Activation('softmax')) 
model.compile(loss='mse', optimizer=Adam(lr=learning_rate), metrics=['mean_squared_error'])
model.build()

model.summary()
        
model.save("//home//nvidia//workspace//nicholasnapolitano//data//MainModel.h5")

with open("//home//nvidia//workspace//nicholasnapolitano//data//Replay_Memory.csv", "wb") as fp:   #Pickling
    pickle.dump(replay_memory, fp)












