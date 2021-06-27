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

# 1 - Normal, 2 - virus, 3 - bacteria
df['type'].replace('Normal', 0, inplace=True)
df['type'].replace('Virus', 1, inplace=True)
df['type'].replace('bacteria', 2, inplace=True) 

unique, counts = np.unique(np.array(df['type']), return_counts=True)
print(dict(zip(unique, counts)))


# In[3]:
img_height, img_width = 128, 128

training_data = []
testing_data = []

def create_data():
    for filename, filetype in zip(df['image_name'], df['type']): 
        img = cv2.imread((train_image_path + filename), cv2.IMREAD_GRAYSCALE)
        new = cv2.resize(img, (img_height, img_width))
        training_data.append([new, filetype])
        
    for filename, fileidx in zip(df_test['image_name'], df_test['id']): 
        img = cv2.imread((test_image_path + filename), cv2.IMREAD_GRAYSCALE)
        new = cv2.resize(img, (img_height, img_width))
        testing_data.append([new, fileidx])


# In[4]:


# create_data()

# import random
# random.shuffle(training_data)

# X = []
# y = []
# X_test = []
# id_test = []

# for features, label in training_data:
#     X.append(features)
#     y.append(label)
    
# for features, idx in testing_data:
#     X_test.append(features)
#     id_test.append(idx)

# X = np.array(X).reshape(-1, img_width, img_height, 1)
# X_test = np.array(X_test).reshape(-1, img_width, img_height, 1)
# y = np.array(y)
# id_test = np.array(id_test)

# with open('./pickles/X_128.pickle', 'wb') as handle:
#     pickle.dump(X, handle)
# with open('./pickles/X_val_128.pickle', 'wb') as handle:
#     pickle.dump(X_test, handle)
# with open('./pickles/y_128.pickle', 'wb') as handle:
#     pickle.dump(y, handle)
# with open('./pickles/y_val_128.pickle', 'wb') as handle:
#     pickle.dump(id_test, handle)


# In[]

with open('./pickles/X_128.pickle', 'rb') as handle:
    X = pickle.load(handle)
with open('./pickles/y_128.pickle', 'rb') as handle:
    y = pickle.load(handle)
with open('./pickles/X_test_128.pickle', 'rb') as handle:
    X_test = pickle.load(handle)
with open('./pickles/y_test_128.pickle', 'rb') as handle:
    y_test = pickle.load(handle)   

# In[]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

from keras.utils import to_categorical
y_ohe = to_categorical(y_train, num_classes=3)
y_val_ohe = to_categorical(y_val, num_classes=3)


train_generator = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    rotation_range=2, # don't want it rotated too much cause what kind of doctor would take a heavily off axis xray?
    width_shift_range=0.1,
    height_shift_range=0.1,
    # horizontal_flip=True,
    shear_range=0.1,
)

test_generator = ImageDataGenerator(
    rescale=1./255
)

train = train_generator.flow(
    x=X_train,
    y=y_ohe,
    # save_to_dir='/media/brendan/860 Evo/Ubuntu/augmentations',
    batch_size=8,
    shuffle=True,
    seed=123
)

val = test_generator.flow(
    x=X_val,
    y=y_val_ohe,
    batch_size=8,
    shuffle=False,
)

test = test_generator.flow(
    x=X_test,
    shuffle=False
)

from sklearn.utils.class_weight import compute_class_weight
y_integers = np.argmax(y_ohe, axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
class_weights = dict(enumerate(class_weights))
print(class_weights)



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
NAME = 'doops_{}'.format(time.time())
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

early = EarlyStopping(monitor='val_loss', mode='min', patience=5)
checkpoint = ModelCheckpoint('/media/brendan/860 Evo/Ubuntu/checkpoints/{}'.format(NAME), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_frequency=1)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.3, min_lr=0.000001)
callbacks_list = [ checkpoint, learning_rate_reduction, early, tensorboard ]


model = Sequential()
model.add(Conv2D(32, 3, padding='SAME', activation='relu', input_shape=X.shape[1:]))
model.add(Conv2D(32, 3, padding='SAME', activation='relu'))
model.add(MaxPool2D(2))
model.add(BatchNormalization())

model.add(Conv2D(64, 3, padding='SAME', activation='relu'))
model.add(Conv2D(64, 3, padding='SAME', activation='relu'))
model.add(Conv2D(64, 3, padding='SAME', activation='relu'))
model.add(MaxPool2D(2))

model.add(Conv2D(128, 3, padding='SAME', activation='relu'))
model.add(Conv2D(128, 3, padding='SAME', activation='relu'))
model.add(Conv2D(128, 3, padding='SAME', activation='relu'))
model.add(MaxPool2D(2))

model.add(Conv2D(256, 3, padding='SAME', activation='relu'))
model.add(Conv2D(256, 3, padding='SAME', activation='relu'))
model.add(Conv2D(256, 3, padding='SAME', activation='relu'))
model.add(MaxPool2D(2))

model.add(Conv2D(512, 3, padding='SAME', activation='relu'))
model.add(MaxPool2D(2))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))

model.add(Dense(3, activation='softmax'))
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()



# In[]


model.fit(train, epochs=100, workers=8, steps_per_epoch=len(train), validation_data=val, validation_steps=len(val), class_weight=class_weights, callbacks=[callbacks_list])
print('Model prediction:')
pred = np.argmax(model.predict(test), axis=-1)
pred = pred+1
print(pred)
get_accuracy(pred)
output = pd.DataFrame({'id': y_test, 'type': pred})
output.to_csv('prediction.csv', index=False)

print('Checkpoint prediction:')
model2 = keras.models.load_model('/media/brendan/860 Evo/Ubuntu/checkpoints/{}'.format(NAME))
pred = np.argmax(model2.predict(test), axis=-1)
model2 = None
pred = pred+1
print(pred)
get_accuracy(pred)
output = pd.DataFrame({'id': y_test, 'type': pred})
output.to_csv('prediction2.csv', index=False)


# In[]
# model = keras.models.load_model('/media/brendan/860 Evo/Ubuntu/checkpoints/{}'.format('1618595033.3271477'))
# NAME = '{}'.format(time.time())

# tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
# early = EarlyStopping(monitor='val_loss', mode='min', patience=5)
# checkpoint = ModelCheckpoint('/media/brendan/860 Evo/Ubuntu/checkpoints/{}'.format(NAME), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_frequency=1)
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.3, min_lr=0.000001)
# callbacks_list = [ checkpoint, learning_rate_reduction, early, tensorboard ]


# for i in range(10):

#     model.fit(train, epochs=5, steps_per_epoch=len(train), validation_data=val, validation_steps=len(val), callbacks=[callbacks_list])
#     pred = np.argmax(model.predict(test), axis=-1)
#     pred = pred+1
#     print(pred)
#     get_accuracy(pred)

#     model2 = keras.models.load_model('/media/brendan/860 Evo/Ubuntu/checkpoints/{}'.format(NAME))
#     pred = np.argmax(model2.predict(test), axis=-1)
#     model2 = None
#     pred = pred+1
#     print(pred)
#     get_accuracy(pred)

# In[]

# model2 = keras.models.load_model('/media/brendan/860 Evo/Ubuntu//{}'.format(0.9006))
# pred = np.argmax(model2.predict(test), axis=-1)
# pred = pred+1
# print(pred)
# output = pd.DataFrame({'id': y_test, 'type': pred})
# output.to_csv('prediction.csv', index=False)

# df_test2 = pd.read_csv('./thing.csv')
# df_test2

# testing_data2 = []
# for ftype in df_test2['type']: 
#     testing_data2.append(ftype)

# num = 0.0
# for i, j in zip(testing_data2, pred):
#     if i == j:
#         num = num + 1
        
# print(num/624)



















# %%
