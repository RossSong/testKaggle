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

import tensorflow as tf
import pandas as pd
from sklearn import metrics
import pickle
import numpy as np
import time
import random

# parameters
LEARNING_RATE = 0.001
DECAY_RATE = 0.9

OUTPUT_SIZE = 2
KERNEL=[3, 2]
NUM_FEATURE=[32, 64, 128, 256, 512]

#initializers_초기화 함수)
initializer = {
    'truncated': tf.truncated_normal_initializer(),
    'xavier': tf.contrib.layers.xavier_initializer(),
    'he': tf.contrib.keras.initializers.he_normal()}

activations = {
    'sigmoid' : tf.nn.sigmoid,
    'relu' : tf.nn.relu,
    'lrelu' : tf.nn.leaky_relu,
'elu':tf.nn.elu}

def weight_variable(shape, init, name):
    return tf.get_variable(shape = shape, initializer=initializer[init], name=name)
def bias_variable(shape, name):
    return tf.get_variable(shape=shape, initializer= tf.zeros_initializer(), name=name)
def max_pool(inputs, kernel, stride):
    return tf.contrib.layers.max_pool2d(
        inputs,
        kernel_size=[kernel, kernel],
        stride=stride,
        padding='SAME',
        outputs_collections=None,
        scope=None)
def batch_normalization(x, training, scope):
    with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm],
                                        scope=scope,
                                        updates_collections=None,
                                        decay=0.9,
                                        center=True,
                                        scale=True,
                                        zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: tf.contrib.layers.batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: tf.contrib.layers.batch_norm(inputs=x, is_training=training, reuse=True))
def conv(inpunt, num_featrue, kernel, act, init, con = None) :
    if con ==None:
        return tf.contrib.layers.conv2d(inpunt,
                             num_outputs = num_featrue,
                             kernel_size = [kernel, kernel],
                             padding = 'SAME',
                             activation_fn = act,
                             weights_initializer = init,
                             biases_initializer = tf.zeros_initializer)
    elif con =='de':
        return tf.contrib.layers.conv2d_transpose(inpunt,
                                                 num_outputs = num_featrue,
                                                 kernel_size = [kernel, kernel],
                                                 padding='SAME',
                                                 activation_fn = act,
                                                 weights_initializer=init,
                                                 biases_initializer=tf.zeros_initializer
                                                 )
def batch_conv(inpunt, num_featrue, kernel, act, init, training, scope) :
    conv = tf.contrib.layers.conv2d(inpunt,
                             num_outputs = num_featrue,
                             kernel_size = [kernel, kernel],
                             padding = 'SAME',
                             activation_fn = None,
                             weights_initializer = init,
                             biases_initializer = tf.zeros_initializer)
    conv = batch_normalization(conv, training, scope)
    conv = act(conv)
    return conv

tf.reset_default_graph()

input_x = tf.placeholder(tf.float32, [None, None, None, None])
input_y = tf.placeholder(tf.float32, [None, None])
training = tf.placeholder(tf.bool, name='training')
keep_prob = tf.placeholder(tf.float32)

# batch normalization + convolutional layers
la_active = activations['elu']
c_initializer = initializer['he']
x_image = tf.reshape(input_x, [-1, 120, 120, 3])
conv_layer1 = batch_conv(x_image, NUM_FEATURE[0], KERNEL[0], la_active, c_initializer, training, 'conv1')
conv_layer2 = batch_conv(conv_layer1, NUM_FEATURE[0], KERNEL[0], la_active, c_initializer, training, 'conv2')
conv_layer3 = batch_conv(conv_layer2, NUM_FEATURE[0], KERNEL[0], la_active, c_initializer, training, 'conv3')
pool1 = max_pool(conv_layer3, KERNEL[0], 2)

conv_layer4 = batch_conv(pool1, NUM_FEATURE[1], KERNEL[1], la_active, c_initializer, training, 'conv4')
conv_layer5 = batch_conv(conv_layer4, NUM_FEATURE[1], KERNEL[1], la_active, c_initializer, training, 'conv5')
conv_layer6 = batch_conv(conv_layer5, NUM_FEATURE[1], KERNEL[1], la_active, c_initializer, training, 'conv6')
pool2 = max_pool(conv_layer6, KERNEL[0], 2)

conv_layer7 = batch_conv(pool2, NUM_FEATURE[2], KERNEL[1], la_active, c_initializer, training, 'conv7')
conv_layer8 = batch_conv(conv_layer7, NUM_FEATURE[2], KERNEL[1], la_active, c_initializer, training, 'conv8')
pool3 = max_pool(conv_layer8, KERNEL[0], 2)

conv_layer9 = batch_conv(pool3, NUM_FEATURE[3], KERNEL[1], la_active, c_initializer, training, 'conv9')
conv_layer10 = batch_conv(conv_layer9, NUM_FEATURE[3], KERNEL[1], la_active, c_initializer, training, 'conv10')
pool5 = max_pool(conv_layer10, KERNEL[0], 2)

layer_shape = pool5.shape
re_img = tf.reshape(pool5, [-1, layer_shape[1]*layer_shape[2]*layer_shape[3]])

fully_layer = tf.contrib.layers.fully_connected(re_img, 2048, activation_fn= tf.nn.elu,
                                                weights_initializer = c_initializer,
                                                biases_initializer = tf.zeros_initializer)
batch_fully = batch_normalization(fully_layer, training, 'fully')

fully_layer1 = tf.contrib.layers.fully_connected(batch_fully, 2048, activation_fn= tf.nn.elu,
                                                weights_initializer = c_initializer,
                                                biases_initializer = tf.zeros_initializer)
batch_fully1 = batch_normalization(fully_layer1, training, 'fully1')

score = tf.contrib.layers.fully_connected(batch_fully1, OUTPUT_SIZE, activation_fn = None)

# Cost to be optimized
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_y, logits=score))

# Optimization method
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

# Performance measures
prediction = tf.argmax(score, 1)
softmax = tf.nn.softmax(logits=score)
correct_prediction = tf.equal(prediction, tf.argmax(input_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

BATCH_SIZE = 64
EPOCH = 500
cnt=0
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    for i in range(EPOCH):
        training_batch = zip(range(0, len(train_x), BATCH_SIZE), range(BATCH_SIZE, len(train_x), BATCH_SIZE))
        for start, end in training_batch:
            cnt+=1
            _, train_cost, train_accuracy = sess.run([optimizer, cost, accuracy],
                                                           feed_dict={input_x: list(train_x)[start:end],
                                                                      input_y: list(pd.get_dummies(train_y).values[start:end]),
                                                                      training: True})

            dev_idx = random.sample(range(len(test_set)), BATCH_SIZE)
            valid_cost, valid_accuracy = sess.run([cost, accuracy],
                                      feed_dict={input_x: list(test_set.x.iloc[dev_idx].values),
                                                 input_y: list(pd.get_dummies(test_set.y[dev_idx]).values),
                                                 training: False})
            end_time = time.time()
            time_dif = end_time - start_time

            print('======================================')
            print("Epoch: {:g}, Step: {:g}".format(i, cnt))
            print("Train loss: {:g}, Train acc: {:g}".format(train_cost, train_accuracy))
            print("Valid loss: {:g}, Valid acc: {:g}".format(valid_cost, valid_accuracy))
            print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

        saver.save(sess, "DogsVsCats("+str(i)+")")

