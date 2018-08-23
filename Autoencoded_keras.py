# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 10:10:45 2018

@author: wh
"""

import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model 
from keras.layers import Dense,Input


(X_train,_),(X_test,Y_test)=mnist.load_data()

X_train=X_train.astype('float32')/255.0-0.5
X_test=X_test.astype('float32')/255.0-0.5
X_train=X_train.reshape(X_train.shape[0],-1)
X_test=X_test.reshape(X_test.shape[0],-1)


print(X_train.shape)
print(X_test.shape)

encoding_dim=2

input_img=Input(shape=(784,))

# 创建encoder网络
encoded=Dense(128,activation='relu')(input_img)
encoded=Dense(64,activation='relu')(encoded)
encoded=Dense(10,activation='relu')(encoded)
encoder_output=Dense(encoding_dim)(encoded)

decoded = Dense(10,activation='relu')(encoder_output)
decoded = Dense(64,activation='relu')(encoded)
decoded = Dense(128,activation='relu')(encoded)
decoded = Dense(784,activation='tanh')(encoded)

autoencoder=Model(input=input_img, output=decoded)

encoder=Model(input=input_img, output=encoder_output)


autoencoder.compile(
        optimizer='adam',
        loss='mse')


print('Training......')
autoencoder.fit(X_train,X_train,epochs=20,batch_size=256,shuffle=True)


print('Testing5555555555555555')
encoded_imgs=encoder.predict(X_test)

print(encoded_imgs.shape)

plt.scatter(encoded_imgs[:,0],encoded_imgs[:,1],c=Y_test)

        
        