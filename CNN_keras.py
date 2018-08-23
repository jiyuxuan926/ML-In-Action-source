# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 10:10:45 2018

@author: wh
"""

import numpy as np
np.random.seed(1337)
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential 
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam


(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

print(X_train.shape)
print(X_train.reshape(X_train.shape[0],-1).shape)

X_train=X_train.reshape(-1,1,28,28)
X_test=X_test.reshape(-1,1,28,28)
Y_train=np_utils.to_categorical(Y_train,num_classes=10)
Y_test= np_utils.to_categorical(Y_test,num_classes=10) 


# 创建神经网络


model = Sequential()
model.add(Convolution2D(32,(5,5),border_mode='same',input_shape=(1,28,28)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode='same'))
print('ssdfyushjkhifdd')
model.add(Convolution2D(64,(5,5),border_mode='same'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))

model.add(Flatten())

model.add(Dense(1024))

model.add(Activation('relu'))

model.add(Dense(10,activation='softmax'))

adam=Adam(lr=1e-4)

model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy'])


print('Training......')
model.fit(X_train,Y_train,epochs=10,batch_size=32)


print('Testing')
loss, accuracy= model.evaluate(X_test,Y_test)

print('test loss:',loss)
print('test accuracy:',accuracy)

        
        