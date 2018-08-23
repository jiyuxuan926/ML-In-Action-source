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
from keras.layers import Dense,Activation
from keras.optimizers import RMSprop


(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

print(X_train.shape)
print(X_train.reshape(X_train.shape[0],-1).shape)

X_train=X_train.reshape(X_train.shape[0],-1)/255
X_test=X_test.reshape(X_test.shape[0],-1)/255
Y_train=np_utils.to_categorical(Y_train,num_classes=10)
Y_test= np_utils.to_categorical(Y_test,num_classes=10) 


# 创建神经网络


model = Sequential([
        Dense(32,input_dim=784),
        Activation('relu'),
        Dense(10),
        Activation('softmax')        
        ])


rmsprop=RMSprop(lr=0.001, rho=0.9,epsilon=1e-08,decay=0.0)
model.compile(
        optimizer=rmsprop,
        loss='categorical_crossentropy',
        metrics=['accuracy'])


print('Training......')
model.fit(X_train,Y_train,epochs=10,batch_size=32)


print('Testing')
loss, accuracy= model.evaluate(X_test,Y_test)
W,b=model.layers[0].get_weights()
print('test loss:',loss)
print('test accuracy:',accuracy)

        
        