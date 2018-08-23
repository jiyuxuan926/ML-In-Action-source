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
from keras.layers import Dense,SimpleRNN
from keras.optimizers import Adam


time_steps=28    #same as the heihgt of the image
input_size=28
batch_size=50
batch_index=0
output_size=10
cell_size=50
lr=0.001


(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

print(X_train.shape)
print(X_train.reshape(X_train.shape[0],-1).shape)

X_train=X_train.reshape(-1,28,28)/255
X_test=X_test.reshape(-1,28,28)/255
Y_train=np_utils.to_categorical(Y_train,num_classes=10)
Y_test= np_utils.to_categorical(Y_test,num_classes=10) 


# 创建神经网络

model = Sequential()
model.add(SimpleRNN(
        batch_input_shape=(batch_size,time_steps,input_size),
        output_dim=cell_size
        ))

model.add(Dense(output_size,activation='softmax'))

adam=Adam(lr)

model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy'])


print('Training......')
# model.fit(X_train,Y_train,epochs=10,batch_size=32)
for step in range(4001):
    X_batch=X_train[batch_index:batch_size+batch_index,:,:]
    Y_batch=Y_train[batch_index:batch_size+batch_index,:]
    cost=model.train_on_batch(X_batch,Y_batch)
    
    batch_index +=batch_size
    batch_index =0 if batch_index >=X_train.shape[0] else batch_index
    
    if step%500==0:
        loss, accuracy= model.evaluate(X_test,Y_test,batch_size=batch_size,
                                       verbose=False)
        print('test loss:',loss)
        print('test accuracy:',accuracy)



        
        