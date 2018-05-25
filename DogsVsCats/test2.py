import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import timedelta
import pickle

train_path = '/Users/ross/testKaggle/DogsVsCats/train/'
test_path = '/Users/ross/testKaggle/DogsVsCats/test1/'

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
        img_array.append(imresize(plt.imread(path+i), (224, 224, 3)))
        cnt+=1
        print(cnt)
    return img_array

label = get_label(train_path)
array = get_array(train_path)

train_x, test_x, train_y, test_y = train_test_split(array, label, test_size=0.2)
test_set = pd.DataFrame({'x': test_x, 'y': test_y})

train_x = np.array(train_x)
test_x = np.array(test_x)

train_x = train_x.reshape(train_x.shape[0], 224, 224, 3)
test_x = test_x.reshape(test_x.shape[0], 224, 224, 3)

print(train_x.shape)
print(test_x.shape)

train_y = pd.get_dummies(train_y).values

import pandas as pd
from sklearn import metrics
import pickle
import numpy as np
import time
import random
from keras import optimizers
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.optimizers import RMSprop
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

input_shape = (224, 224, 3)
BATCH_SIZE = 16
EPOCH = 500

model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
           activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(2, activation='sigmoid')
])

#model.compile(optimizer=RMSprop(lr=0.001, rho=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Fit the model
model.fit(train_x, train_y, validation_split=0.2, epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=callbacks_list, verbose=1)
#model.fit(train_x, train_y, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1)

evaluation = model.evaluate(test_x, test_y, verbose=1)
print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))
