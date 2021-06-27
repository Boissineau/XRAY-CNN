#!/usr/bin/env python
# coding: utf-8

# In[1]:


from random import shuffle
import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, Dense, Activation
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import gc
import time
import sys
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.engine.data_adapter import expand_1d, train_validation_split

from tensorflow.python.keras.engine.sequential import relax_input_shape
from tensorflow.python.keras.layers.convolutional import Conv

import kerastuner
from kerastuner.tuners import RandomSearch, Hyperband
from kerastuner.engine.hyperparameters import HyperParameters



# In[2]:

from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)

train_image_path = './images/images/train2/'
test_image_path = './images/images/test/'

train_csv = './assignment5_training_data_metadata3.csv'
test_csv = './assignment5_test_data_metadata.csv'
df = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)



# In[3]:
img_height, img_width = 512, 512

training_data = []
testing_data = []


train_generator = ImageDataGenerator(
    rescale=1./255,
    # zoom_range=0.2,
    rotation_range=2,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    shear_range=0.1
)

test_generator = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1
)

train = train_generator.flow_from_directory(
    '/home/brendan/projects/algo/images/images/second_batch_train',
    target_size=(128,128),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32,
    seed=123,
    shuffle=True,
)

val = test_generator.flow_from_directory(
    '/home/brendan/projects/algo/images/images/second_batch_val',
    target_size=(128,128),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32,
    # shuffle=True,
    seed=123,
)

test = test_generator.flow_from_directory(
    '/home/brendan/projects/algo/images/images/second_batch_test',
    target_size=(128,128),
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False,
)

# In[]
def get_accuracy(pred):
    df_test2 = pd.read_csv('./thing.csv')
    df_test2

    testing_data2 = []
    for ftype in df_test2['type']: 
        testing_data2.append(ftype)

    num = 0.0
    for i, j in zip(testing_data2, pred):
        if i == j:
            num = num + 1
            
    print(num/624)

# In[]


opt = keras.optimizers.Adam(lr=0.00003)
# opt = keras.optimizers.Nadam(lr=0.0003)
NAME = 'doops_{}'.format(time.time())
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

early = EarlyStopping(monitor='val_loss', mode='min', patience=5)
checkpoint = ModelCheckpoint('/media/brendan/860 Evo/Ubuntu/checkpoints/{}'.format(NAME), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_frequency=1)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.3, min_lr=0.000001)
callbacks_list = [ checkpoint, learning_rate_reduction, early, tensorboard ]

model = Sequential()
model.add(Conv2D(32, 3, padding='SAME', activation='relu', input_shape=train.image_shape))
model.add(MaxPool2D(2))

model.add(Conv2D(32, 3, padding='SAME', activation='relu'))
model.add(Conv2D(32, 3, padding='SAME', activation='relu'))
model.add(Conv2D(32, 3, padding='SAME', activation='relu'))
model.add(MaxPool2D(2))

model.add(Conv2D(64, 3, padding='SAME', activation='relu'))
model.add(MaxPool2D(2))

model.add(Conv2D(128, 3, padding='SAME', activation='relu'))
model.add(MaxPool2D(2))

model.add(Conv2D(128, 3, padding='SAME', activation='relu'))
model.add(MaxPool2D(2))

model.add(Conv2D(256, 3, padding='SAME', activation='relu'))
model.add(MaxPool2D(2))


model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(train, epochs=100, steps_per_epoch=len(train), validation_data=val, validation_steps=len(val), workers=16, callbacks=callbacks_list)



# In[]

def get_accuracy(pred):
    df_test2 = pd.read_csv('./thing.csv')
    df_test2

    testing_data2 = []
    for ftype in df_test2['type']: 
        testing_data2.append(ftype)

    num = 0.0
    for i, j in zip(testing_data2, pred):
        if i == j:
            num = num + 1
            
    print(num/624)

print('Model prediction:')
pred = np.argmax(model.predict(test), axis=-1)
pred = pred+1
print(pred)
get_accuracy(pred)





















# %%
