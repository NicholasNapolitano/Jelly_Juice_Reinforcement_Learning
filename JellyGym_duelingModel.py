#Import Librerie

import csv
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

#Creazione Replay Memory

replay_memory = []

#Inizializzazione learning rate

learning_rate = 0.001

#Creazione Modello

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
model.compile(loss='mse', optimizer=Adam(lr=learning_rate), 
              metrics=['mean_squared_error'])
model.build()

#Stampa Informazioni sul modello

model.summary()

#Aggiornamento dei pesi copiandoli dalla rete principale

mainModel = load_model("MainModel.h5")
model.set_weights(mainModel.get_weights())

#Salvataggio del Modello

model.save("DuelingModel.h5")
