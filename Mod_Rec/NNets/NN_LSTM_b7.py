#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Devoloper:  Tina Burns
School: Rutger University
Advisor: Rchard Martin
Python Version: Python3

Updated from r3
"""
#%% Imports necessary libraries
import numpy as np, os, pandas as pd, time
import keras
from keras.layers import Dropout, Dense, LSTM
from keras import backend as K 
from keras.models import Sequential
from tensorflow.keras.models import load_model
#Sets up things for the environment
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TENSORFLOW_FLAGS"] = "device=gpu%d" % (0)

from numpy.random import seed
seed(1337)
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
        # clears preious keras nn session
        K.clear_session()
    def getType(self):
        return "LSTM"    
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
    
    #Main NN Execution
    def runNN(self, X_train, Y_train, X_test, Y_test, X_val, Y_val, 
              Y_train_label= None, Y_test_label = None, 
              Y_val_label = None, act1 = "relu",  act2 = "softmax", 
              epochs = 10, batch_size = 1024, testAct = False, mod = '',
              train_model = True, folder_NN_hist = "NN_Hist"):
      
        if not testAct:
            act1 = "sofplus"  
            act2 = "softmax"
          
        weight_file = os.path.join(folder_NN_hist, "LSTM_weights.h5"); 
        model_file = os.path.join(folder_NN_hist, "LSTM_model"); 
        hist_file = os.path.join(folder_NN_hist, "LSTM_history.csv");
        
        time_train_start = time.time()
        if train_model: 
            #Removes previous files to prevent file creation errors
            if not os.path.exists(folder_NN_hist): os.makedirs(folder_NN_hist)    
            if os.path.exists(weight_file): os.remove(weight_file)
            if os.path.exists(model_file): os.remove(model_file)
            if os.path.exists(hist_file): os.remove(hist_file)
            model = Sequential()
            model.add(LSTM(32, return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2]))) 
            #model.add(LSTM(32, return_sequences=True))
            model.add(LSTM(32)) 
            model.add(Dropout(0.2)) 
            model.add(Dense(Y_train.shape[1], activation= act2))
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            print(model.summary())
            hist = model.fit(X_train, Y_train, 
                                validation_data=(X_val, Y_val), epochs=epochs, batch_size=1024)
    
            loss_val_train = np.asarray(hist.history['val_loss'])
            acc_val_train = np.asarray(hist.history['val_accuracy'])
            pd.DataFrame({"loss_val_train" : loss_val_train,
                           "acc_val_train" : acc_val_train,
                            }).to_csv(hist_file)
            model.save_weights(weight_file)
            model.save(model_file)
        else:
            model = load_model(model_file)
            #model()
            hist = pd.read_csv(hist_file)
            #print(hist)
            loss_val_train = hist["loss_val_train"].values
            acc_val_train = hist["acc_val_train"].values                    
        time_train = np.round(time.time() - time_train_start, 2)

        time_test_start = time.time()
        score = model.evaluate(X_test, Y_test, batch_size=64)
        pred = model.predict(X_test, verbose = 1)
        time_test = np.round(time.time() - time_test_start, 2)

        print("Train Data Shape: ", X_train.shape)
        print("Accuracy ", score[1])
        print("Loss: ", score[0])
        print("NN Time: ", time_train + time_test, '\n')
        #Validation loss is taken as the final value in the array of validation loss in the training data
        #Returns Test loss, test accuracy, validation loss, validation accuracy 
        return (score[0], score[1], loss_val_train[-1], acc_val_train[-1], 
                pred, act1, act2, time_train, time_test)
# %% Execution
def test():
    NNet = NN()
    NNet.__init__
    x, y = generateSequence(1000, 100)    
    x = NNet.shuffleData(x)
    y = NNet.shuffleData(y)
    y = keras.utils.to_categorical(y)

    #This shapes the array so that it is evenly divisible by 4
    #which allows it to be correctly processed by the nueral network
    x = x[:, 0:(x.shape[1] - x.shape[1]%4)]

    # Reshapes the data so that it is 4 dimensional
    # Seperates test and data into training, test, and validiation sets
    x = x.reshape(-1, x.shape[1], 1) / np.max(x)
    #y = y.reshape(-1, 1, y.shape[1], 1) / np.max(y)
    X_train, X_test, X_val = NNet.genTrainTest(x)
    Y_train, Y_test, Y_val = NNet.genTrainTest(y)
    
    NNet.runNN(X_train, Y_train, X_test, Y_test, X_val, Y_val,epochs=10, 
               train_model = True)
    return 0

#%%
if __name__ == '__main__':
    test()


    