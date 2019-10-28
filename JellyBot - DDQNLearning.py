import time
import random
import numpy as np
import csv
import pickle
from numpy import loadtxt
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import tensorflow as tf
import os

#Utile per ridurre il carico sulla CPU
NUM_PARALLEL_EXEC_UNITS = 16
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,
                       allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
session = tf.compat.v1.Session(config=config)
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

validMoves = loadtxt('validMoves.txt')
mossa = int(loadtxt('prevOutput.txt'))
gamma = 0.95

elem = []

model = load_model("MainModel.h5")
target_model = load_model("DuelingModel.h5")

with open("Replay_Memory.csv", "rb") as fp:   # Unpickling
     replay_memory = pickle.load(fp)

batch_size = int(0.1*len(replay_memory))

state = loadtxt('prev_board_state.csv', delimiter=',')

state = state.tolist()
target = state.pop()
loop = 81 - len(state)
for i in range(loop):
    state.append(0)
state.append(target)
state = np.array(state)
state = state.reshape(-1, 82)
print(state)

elem.append(state)
 
elem.append(mossa)

reward = int(loadtxt("reward.txt'))

elem.append(reward)

next_state = loadtxt('board_state.csv', delimiter=',')
                     
next_state = next_state.tolist()
next_target = next_state.pop()
next_loop = 81 - len(next_state)
for i in range(next_loop):
    next_state.append(0)
next_state.append(next_target)
next_state = np.array(next_state)
next_state = next_state.reshape(-1, 82)
                     
elem.append(next_state)

index = mossa

mossa = to_categorical([mossa], 324)

listOfLines = list()        
with open ("possibleMoves.txt", "r") as myfile:
    for line in myfile:
        listOfLines.append(line.strip())


if(reward >= 0.1):
    for i in range(len(mossa[0])):
        mossa[0][i] = -1
    mossa[0][index] = reward
    #print(mossa)
    model.fit(state, mossa, epochs=1000, verbose=1)
    if(len(replay_memory) < 501): 
        replay_memory.append(tuple(elem))
    else:
        replay_memory.pop(0)
        replay_memory.append(tuple(elem))
else:
    for i in range(len(mossa[0])):
        mossa[0][i] = 0
    mossa[0][index] = reward
    model.fit(state, mossa, epochs=1000, verbose=1)


if len(replay_memory) > batch_size:
    minibatch = random.sample(replay_memory, batch_size)
    for state, action, reward, next_state in minibatch:
        time.sleep(0.1)
        target = model.predict(state)
        t = target_model.predict(next_state)[0]
        target[0][action] = reward*(1-0.01*validMoves) + gamma * np.amax(t)
        model.fit(state, target, epochs=100, verbose=1)
      
act_prob = []

for k in range(324):
    index = k
    if(k not in listOfLines):
        index = to_categorical([index], 324)
        for i in range(len(index[0])):
            index[0][i] = -1
        model.fit(next_state, index, epochs=50, verbose=1) 
    
act_values = model.predict(next_state)[0]

for k in range(len(listOfLines)):
    index = int(listOfLines[k])
    act_prob.append(act_values[index])

act_pr = []

for i in range(len(act_prob)):
    act_pr.append(act_prob[i]/sum(act_prob))

somma = 0
for i in range(len(act_pr)):
    somma += act_pr[i]
    
action = np.random.choice(listOfLines, p=act_pr)

print(action)

model.save("MainModel.h5")

with open("Replay_Memory.csv", "wb") as fp:   #Pickling
    pickle.dump(replay_memory, fp)

with open("//home//nvidia//workspace//nicholasnapolitano//data//output.txt", 'w') as f:
        f.write("%s\n" % action)
