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

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import time
import sys
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.engine.data_adapter import expand_1d, train_validation_split

from tensorflow.python.keras.engine.sequential import relax_input_shape
from tensorflow.python.keras.layers.convolutional import Conv


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
df['type'].fillna('Normal', inplace=True)
df['type'].replace('Normal', 0, inplace=True)
df['type'].replace('Virus', 1, inplace=True)
df['type'].replace('bacteria', 2, inplace=True) 

unique, counts = np.unique(np.array(df['type']), return_counts=True)
print(dict(zip(unique, counts)))


# In[3]:


img_height, img_width = 256, 256

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
# # random.shuffle(testing_data)

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

# with open('./pickles/X4.pickle', 'wb') as handle:
#     pickle.dump(X, handle)
# with open('./pickles/y4.pickle', 'wb') as handle:
#     pickle.dump(y, handle)
# with open('./pickles/X_test4.pickle', 'wb') as handle:
#     pickle.dump(X_test, handle)
# with open('./pickles/id_test4.pickle', 'wb') as handle:
#     pickle.dump(id_test, handle)



with open('./pickles/X4.pickle', 'rb') as handle:
    X = pickle.load(handle)
with open('./pickles/y4.pickle', 'rb') as handle:
    y = pickle.load(handle)
with open('./pickles/X_test4.pickle', 'rb') as handle:
    X_test = pickle.load(handle)
with open('./pickles/id_test4.pickle', 'rb') as handle:
    id_test = pickle.load(handle)   


# In[]

# ohe = preprocessing.OneHotEncoder()
# categories = np.array(y.reshape(-1,1))
# y_ohe = ohe.fit_transform(categories).todense()
# y_1 = np.array(y_ohe)

# In[]
from keras.utils import to_categorical
y_ex = to_categorical(y, num_classes=3)



# In[]



image_gen = ImageDataGenerator(
    rescale=1./255,
    # shear_range=0.2,
    zoom_range=0.2,
    rotation_range=15,
    brightness_range=[0.8,1.4],
    width_shift_range=0.2,
    height_shift_range=0.2,
    # horizontal_flip=True,
    validation_split=0.1,

)

test_data_gen = ImageDataGenerator(rescale = 1./255)

train = image_gen.flow(
    x=X,
    y=y_ex,
    subset='training',
    batch_size=16,
    shuffle=True,
    seed=123
)

'''
VAL SHOULD NOT! BE AUGMENTED! 
'''
val = image_gen.flow(
    x=X,
    y=y_ex,
    subset='validation',
    batch_size=16,
    shuffle=True,
    seed=123
)


test = test_data_gen.flow(
      x=X_test,
      shuffle=False, 
)


from sklearn.utils.class_weight import compute_class_weight
y_integers = np.argmax(y_ex, axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
class_weights = dict(enumerate(class_weights))
# In[10]
model = Sequential()
NAME = '{}'.format(time.time())
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

model.add(Conv2D(32, 3, activation='relu', input_shape=X.shape[1:]))
model.add(MaxPool2D(2))
model.add(Conv2D(64, 3, activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPool2D(2))
model.add(Conv2D(128, 3, activation='relu'))
model.add(MaxPool2D(2))
model.add(Conv2D(256, 3, activation='relu'))

model.add(Flatten())
# model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))

model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()



# early = EarlyStopping(monitor='val_loss', mode='min', patience=5)
checkpoint = ModelCheckpoint('/media/brendan/860 Evo/Ubuntu/checkpoints/{}'.format(NAME), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_frequency=1)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.3, min_lr=0.0001)
callbacks_list = [ checkpoint, learning_rate_reduction, tensorboard ]
history = model.fit(train, epochs=25, validation_data=val, class_weight=class_weights, callbacks=[callbacks_list])

model2 = keras.models.load_model('/media/brendan/860 Evo/Ubuntu/checkpoints/{}'.format(NAME))
pred = np.argmax(model2.predict(test), axis=-1)
pred = pred+1
output = pd.DataFrame({'id': id_test, 'type': pred})
output.to_csv('prediction.csv', index=False)


df_test2 = pd.read_csv('./thing.csv')
df_test2

testing_data2 = []
for ftype in df_test2['type']: 
    testing_data2.append(ftype)

num = 0.0
for i, j in zip(testing_data2, pred):
    if i == j:
        num = num + 1
model2.save('/media/brendan/860 Evo/Ubuntu/{:.4f}'.format(num/624))
print(num/624)

# In[]

# In[]

# model2 = keras.models.load_model('/media/brendan/860 Evo/Ubuntu/0.9087')
# NAME = '0.9087_testing_{}'.format(time.time())
# tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# checkpoint = ModelCheckpoint('/media/brendan/860 Evo/Ubuntu/checkpoints/{}'.format(NAME), monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_frequency=1)
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.3, min_lr=0.00001)
# callbacks_list = [ checkpoint, learning_rate_reduction, tensorboard ]
# model2.fit(train, epochs=25, validation_data=val2, class_weight=d_class_weights, callbacks=[callbacks_list])


# model3 = keras.models.load_model('/media/brendan/860 Evo/Ubuntu/checkpoints/{}'.format(NAME))
# pred = np.argmax(model3.predict(test), axis=-1)
# pred = pred+1
# output = pd.DataFrame({'id': id_test, 'type': pred})
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
# model3.save('/media/brendan/860 Evo/Ubuntu/{:.4f}'.format(num/624))

# In[]
# pred = np.argmax(model.predict(test), axis=-1)
# pred = pred+1
# output = pd.DataFrame({'id': id_test, 'type': pred})
# output.to_csv('prediction.csv', index=False)



# In[]
# model.save('/media/brendan/860 Evo/Ubuntu/0.88301')
# model = keras.models.load_model('/media/brendan/860 Evo/Ubuntu/best/0.88301')
# model.summary()
# checkpoint = ModelCheckpoint('/media/brendan/860 Evo/Ubuntu/checkpoints/testing', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_frequency=1)
# tensorboard = TensorBoard(log_dir='logs/testing_best_model')
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.3, min_lr=0.000001)
# callbacks_list = [ checkpoint, learning_rate_reduction, tensorboard ]
# model.fit(train, epochs=25, validation_data=val, class_weight=d_class_weights, callbacks=[callbacks_list])


# In[]


# df_test2 = pd.read_csv('./assignment5_training_data_metadata3.csv')

# testing_data2 = []
# for filename, ftype in zip(df_test2['image_name'], df_test2['type']): 
#     img = cv2.imread((test_image_path + filename), cv2.IMREAD_GRAYSCALE)
#     new = cv2.resize(img, (100, 100))
#     testing_data2.append([new, ftype])

# X_2 = []
# y_2 = []
# for img, ftype in testing_data2:
#     X_2.append(img)
#     y_2.append(ftype)

# X_2 = np.array(X_2).reshape(-1, 512, 512, 1)
# y_2 = np.array(y_2)



# with open('./pickles/X_100.pickle', 'wb') as handle:
#     pickle.dump(X_2, handle)
# with open('./pickles/y_100.pickle', 'wb') as handle:
#     pickle.dump(y_2, handle)


# In[]


# In[]

# with open('./pickles/X_data.pickle', 'rb') as handle:
#     X_2 = pickle.load(handle)

# with open('./pickles/y_data.pickle', 'rb') as handle:
#     y_2 = pickle.load(handle)  




# In[]

# ohe = preprocessing.OneHotEncoder()
# categories = np.array(y_2.reshape(-1,1))
# y_ohe2 = ohe.fit_transform(categories).todense()
# y_val = np.array(y_ohe2)



# val2 = image_gen.flow(
#     x=X_2,
#     y=y_val,
#     batch_size=32,
# )










































# %%
