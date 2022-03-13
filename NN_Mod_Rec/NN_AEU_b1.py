#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:43:25 2019
@author: tina-mac2

Devoloper:  Tina Burns
School: Rutger University
Advisor: Rchard Martin
Python Version: Python3
Code Revision: R3

This code for an Unsperivsed Autencoder based on Keras 
"""
# %%
#Imports necessary libraries 
import os, keras, numpy as np, time, pandas as pd
from keras.layers import Dense, Flatten, Dropout
from keras import backend as K 
from keras.layers import Input
from keras.models import Model
from keras.optimizers import RMSprop
from tensorflow.keras.models import load_model

#from keras.utils import multi_gpu_model
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TENSORFLOW_FLAGS"] = "device=gpu%d" % (0)
np.random.seed(1200)  # For reproducibility

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# %% Generates data
#Creates two matrix arrays: One in ascending order and
#One array in descending order
#Return the concantenation of those two arrays and another array for labels
def generateSequence(m, n):
    # return [randint(0, 4) for _ in range(length)]
    arr = np.empty([0, n])
    for i in range(1, m + 1):
        arr = np.append(arr, [np.arange(i, n + i)], axis=0)
    arr = np.concatenate((arr, np.flip(arr)))
    lab = np.concatenate((np.full((m, 1), 0), np.full((m, 1), 1)))
    # print(arr1.shape)
    return (arr, lab)

# %% Neural Network Class
# This neural network is fully connected Sequential network using the keras library
class NN():
    def __init__(self):
        """"""
        # clears preious keras nn session
        K.clear_session()
        
    def getType(self):
        return "AE"
        
    # This function splits the array into three seperate arrays
    def genTrainTest(self, arr):
        sep = int(arr.shape[0] / 5)
        return np.split(arr, [sep * 3, sep * 4], axis=0)
    # Shuffles Data
    def shuffleData(self, x):
        np.random.seed(1200)
        myPermutation = np.random.permutation(x.shape[0])
        x = x[myPermutation]
        return x
    
    # Model definition
    # encoder
    def NN_autoecoder(self, hidden, act1 = "relu", act2 = "softsign", numClasses = 2):
        #Encoder
        hidden = Flatten()(hidden)
        hidden = Dense(256, activation= act1)(hidden)
        hidden = Dense(128, activation= act1)(hidden)
        hidden = Dense(64, activation = act1)(hidden)
        hidden = Dense(32, activation = act1)(hidden)

        #Decoder
        #hidden1 = Flatten()(hidden1)
        hidden = Dense(32, activation= act1)(hidden)
        hidden = Dense(64, activation=  act1)(hidden)
        hidden = Dense(128, activation= act1)(hidden)
        hidden = Dense(256, activation= act1)(hidden)
        #dropout and final dense layer        
        hidden = Dropout(0.2)(hidden)
        hidden = Dense(numClasses, activation= act2)(hidden)
        return hidden
                
    def runNN(self, X_train, Y_train, X_test, Y_test, X_val, Y_val, 
              Y_train_label= None, Y_test_label = None, 
              Y_val_label = None, act1 = "relu",  act2 = "softmax", 
              epochs = 10, batch_size = 128, testAct = False, mod = '',
              train_model = True, folder_NN_hist = "NN_Hist"):   
     
        # weight_file = os.path.join(folder_NN_hist, "AE_weights.h5").replace(r'\'', '/'); 
        # model_file = os.path.join(folder_NN_hist, "AE_model").replace(r'\'', '/'); 
        # hist_file = os.path.join(folder_NN_hist, "AE_history.csv").replace(r'\'', '/');
        
        weight_file = os.path.join(folder_NN_hist, "AE_weights.h5").replace(r'\'', '/'); 
        model_file = os.path.join(folder_NN_hist, "AE_model").replace(r'\'', '/'); 
        hist_file = os.path.join(folder_NN_hist, "AE_history.csv").replace(r'\'', '/');
        
        if not testAct: act1 = "elu"; act2 = "softmax" 

        #numClasses = Y_train.shape[1]
        inChannel = 1
        input_img = Input(shape=(1, X_train.shape[2], inChannel))  
         
        time_train_start = time.time()

        if train_model:
            #Removes previous files to prevent file creation errors
            if not os.path.exists(folder_NN_hist): os.makedirs(folder_NN_hist)
            if os.path.exists(weight_file): os.remove(weight_file)
            if os.path.exists(model_file): os.remove(model_file)
            if os.path.exists(hist_file): os.remove(hist_file)
            
            output = self.NN_autoecoder(input_img, numClasses = Y_train.shape[1], 
                   act1 = act1, act2 = act2)
            
            model = Model(inputs = input_img, outputs = output)
            print(model.summary()) 
            
            model.compile(optimizer='Adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
       
            hist = model.fit(X_train, Y_train,
                              epochs=epochs, batch_size=batch_size, 
                              validation_data=(X_val, Y_val),
                              shuffle=False) 
            model.save_weights(weight_file)
            model.save(model_file)

        
            #loss_test_train = np.asarray(hist.history['loss'])
            #acc_test_train = np.asarray(hist.history['accuracy'])
            loss_val_train = np.asarray(hist.history['val_loss'])
            acc_val_train = np.asarray(hist.history['val_accuracy'])
            pd.DataFrame({"loss_val_train" : loss_val_train,
                           "acc_val_train" : acc_val_train,
                            }).to_csv(hist_file)

        else:
            model = load_model(model_file)
            #model()
            hist = pd.read_csv(hist_file)
            #print(hist)
            loss_val_train = hist["loss_val_train"].values
            acc_val_train = hist["acc_val_train"].values                    
        time_train = np.round(time.time() - time_train_start, 2)

        time_test_start = time.time()
        score = model.evaluate(X_test, Y_test, verbose=1)
        #Gets and outputs predciton of each class
        pred = model.predict(X_test, verbose = 1)
        time_test = np.round(time.time() - time_test_start, 2)

        #print("Data shape: ", x.shape)
        print("Train Data Shape: ", X_train.shape)
        print("Accuracy ", score[1])
        print("Loss: ", score[0], '\n')
        #print("Validation Loss: ", loss_val_train)
        #print("Validation Accuracy: ", acc_val_train)
        #Validation loss is taken as the final value in the array of validation loss in the training data
        #Returns Test loss, test accuracy, validation loss, validation accuracy 
        return (score[0], float(score[1]), float(loss_val_train[0]), 
                float(acc_val_train[0]), pred, act1, act2, time_train, time_test)  
    


#%%    
def test():
    NNet = NN()
    NNet.__init__
    x, y = generateSequence(100, 300)
    x = x.reshape(-1, 1, x.shape[1], 1)
    print(x.shape)
    x = NNet.shuffleData(x)
    y = NNet.shuffleData(y)
    y = keras.utils.to_categorical(y)
    a, b, c = NNet.genTrainTest(x)
    a1, b1, c1 = NNet.genTrainTest(y)
    #NNet.runAutoencoder(a, a, b, b)
    NNet.runNN(a, a1, b, b1, c, c1, train_model = True)

#%% Testing Autoencoder alone
if __name__ == '__main__':
    #pred = test()
    test()

