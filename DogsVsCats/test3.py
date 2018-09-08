import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import timedelta
import pickle

train_path = '.\\data\\train\\'
test_path = '.\\data\\test1\\'

label_pickle = 'label.pickle'
array_pickle = 'array.pickle'

def get_label(path):
    labels = []
    for i in os.listdir(path):
        if 'cat' in i:
            labels.append(0)
        else:
            labels.append(1)
    return labels

def get_array(path):
    cnt = 0
    img_array = []
    for i in os.listdir(path):
        img_array.append(imresize(plt.imread(path+i), (120, 120, 3)))
        cnt+=1
        print(cnt)
    return img_array

def saveLabel():
    label = get_label(train_path)
    with open(label_pickle, 'wb') as f:
        pickle.dump(label, f, pickle.HIGHEST_PROTOCOL)

def saveArray():
    array = get_array(train_path)
    with open(array_pickle, 'wb') as f:
        pickle.dump(array, f, pickle.HIGHEST_PROTOCOL)

def loadLabel():
    if False == os.path.exists(label_pickle):
        saveLabel()

    label = []
    with open('label.pickle', 'rb') as f:
        label = pickle.load(f)

    return label

def loadArray():
    if False == os.path.exists(array_pickle):
        saveArray()

    array = []
    with open('array.pickle', 'rb') as f:
        array = pickle.load(f)
    
    return array

label = loadLabel()
array = loadArray()

train_x, test_x, train_y, test_y = train_test_split(array, label, test_size=0.2)
test_set = pd.DataFrame({'x': test_x, 'y': test_y})

train_x = np.array(train_x)
test_x = np.array(test_x)

train_x = train_x.reshape(train_x.shape[0], 120, 120, 3)
test_x = test_x.reshape(test_x.shape[0], 120, 120, 3)

print(train_x.shape)
print(test_x.shape)
#plt.imshow(train_x[0])
#plt.show()


train_y = pd.get_dummies(train_y).values
#print(train_y[0])

import pandas as pd
from sklearn import metrics
import pickle
import numpy as np
import time
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten
from keras import optimizers
from keras.optimizers import RMSprop
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

input_shape = (120, 120, 3)
BATCH_SIZE = 128
EPOCH = 500

base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(2, activation='sigmoid'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.summary()
model.fit(train_x, train_y, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1)

evaluation = model.evaluate(test_x, test_y, verbose=1)
print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))
