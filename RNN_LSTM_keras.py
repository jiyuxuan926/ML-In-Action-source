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
from keras.layers import Dense,TimeDistributed,LSTM
from keras.optimizers import Adam
import matplotlib.pyplot as plt


batch_start=0    
time_steps=20
batch_size=50
input_size=1
output_size=1
cell_size=20
lr=0.006
batch_index=0

(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

print(X_train.shape)
print(X_train.reshape(X_train.shape[0],-1).shape)

X_train=X_train.reshape(-1,28,28)/255
X_test=X_test.reshape(-1,28,28)/255
Y_train=np_utils.to_categorical(Y_train,num_classes=10)
Y_test= np_utils.to_categorical(Y_test,num_classes=10) 


# 创建神经网络

model = Sequential()
model.add(LSTM(
        batch_input_shape=(batch_size,time_steps,input_size),
        output_dim=cell_size,
        return_sequences=True,
        stateful=True,
        
        ))

model.add(TimeDistributed(Dense(output_size)))

adam=Adam(lr)

model.compile(
        optimizer=adam,
        loss='mse',
        metrics=['accuracy'])


print('Training......')
# model.fit(X_train,Y_train,epochs=10,batch_size=32)
for step in range(4001):
    # X_batch,Y_batch,xs=get_batch()
    X_batch=X_train[batch_index:batch_size+batch_index,:,:]
    Y_batch=Y_train[batch_index:batch_size+batch_index,:]
    cost=model.train_on_batch(X_batch,Y_batch)
    pred=model.predict(X_batch,batch_size)
    '''
    #plt.plot(xs[0,:],Y_batch[0].flatten(),'r',xs[0,:],pred.flatten()[:time_steps],'b--')
    plt.ylim((-1.2,1.2)) 
    plt.draw()
    plt.pause(0.5)
    
    if step%10==0:
        print('train cost:',cost)
'''

        