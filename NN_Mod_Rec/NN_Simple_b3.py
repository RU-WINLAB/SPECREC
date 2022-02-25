#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:43:25 2019
@author: tina-mac2

Devoloper:  Tina Burns
School: Rutger University
Advisor: Rchard Martin
Python Version: Python3
Code Revision: R1

This code runs a categorical keras network.  
"""

# %%
#Imports necessary libraries
import numpy as np, os, pandas as pd, time
import keras
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras import backend as K 
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TENSORFLOW_FLAGS"] = "device=gpu%d" % (0)
from tensorflow.keras.models import load_model
from numpy.random import seed
seed(1337)
#import tensorflow
#tensorflow.random.set_seed(2)
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

#%% Neural Network Class
class NN():
    def __init__(self):
        """"""
        # clears preious keras nn session
        K.clear_session()
    
    def getType(self):
        return "SIMPLE"
        
    # This function splits the array into three seperate arrays
    def genTrainTest(self, arr, axis=0):
        sep = int(arr.shape[0] / 5)
        return np.split(arr, [sep * 3, sep * 4], axis=axis)
   
    # Shuffles Data
    def shuffleData(self, x):
        np.random.seed(1200)
        myPermutation = np.random.permutation(x.shape[0])
        x = x[myPermutation]
        return x
    
    #Main Neural Network
    def NN_FC(self, data, act1 = "softsign", act2 = "relu", numClasses = 2):   
        hidden1 = Flatten()(data)
        hidden1 = Dense(32, activation= act1)(hidden1)
        #hidden1 = Dropout(0.2)(hidden1)
        hidden1 = Dense(numClasses, activation=act2)(hidden1)
        return hidden1

    #Main NN Execution
    def runNN(self, X_train, Y_train, X_test, Y_test, X_val, Y_val, 
              Y_train_label= None, Y_test_label = None, 
              Y_val_label = None, act1 = "relu",  act2 = "softmax", 
              epochs = 10, batch_size = 128, testAct = False, mod = '',
              train_model = True, folder_NN_hist = "NN_Hist" ):
        
        if not os.path.exists(folder_NN_hist): os.makedirs(folder_NN_hist)      
        weight_file = os.path.join(folder_NN_hist, "SIMPLE_weights.h5"); 
        model_file = os.path.join(folder_NN_hist, "SIMPLE_model"); 
        hist_file = os.path.join(folder_NN_hist, "SIMPLE_history.csv");
        
        #Sets up the input shape for the model.  
        #These dimensions are based a on a 4D input where the parameter
        #at index 2 refers the number of datapoints i.e. the number of columns
        inChannel = 1
        input_img = Input(shape=(X_train.shape[1], X_train.shape[2], inChannel)) 
        
        if not testAct:
            ""
            act1 = "relu"  
            act2 = "softmax"

        time_train_start = time.time()
        if train_model:
            #Removes previous files to prevent file creation errors
            if not os.path.exists(folder_NN_hist): os.makedirs(folder_NN_hist)
            if os.path.exists(weight_file): os.remove(weight_file)
            if os.path.exists(model_file): os.remove(model_file)
            if os.path.exists(hist_file): os.remove(hist_file)
            output = self.NN_FC(input_img, numClasses = Y_train.shape[1], 
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
        return (score[0], score[1], loss_val_train[0], acc_val_train[0], 
                pred, act1, act2, time_train, time_test)  

# %% Execution
def test():
    NNet = NN()
    NNet.__init__
    x, y = generateSequence(40, 40)    
    x = NNet.shuffleData(x)
    y = NNet.shuffleData(y)
    y = keras.utils.to_categorical(y)

    #This shapes the array so that it is evenly divisible by 4
    #which allows it to be correctly processed by the nueral network
    x = x[:, 0:(x.shape[1] - x.shape[1]%4)]

    # Reshapes the data so that it is 4 dimensional
    # Seperates test and data into training, test, and validiation sets
    x = x.reshape(-1, 1, x.shape[1], 1) / np.max(x)
    #y = y.reshape(-1, 1, y.shape[1], 1) / np.max(y)
    X_train, X_test, X_val = NNet.genTrainTest(x)
    Y_train, Y_test, Y_val = NNet.genTrainTest(y)
    
    NNet.runNN(X_train, Y_train, X_test, Y_test, X_val, Y_val,epochs=10)
    #print(pred)
    # return acc, loss, pred
    return 0

if __name__ == '__main__':
    test()
   

    