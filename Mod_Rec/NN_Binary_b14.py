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

This code run a Autencoder base on Keras 
"""
# %%
#Imports necessary libraries 
import os, numpy as np, pandas as pd, time
from keras.layers import Dense, Flatten, Dropout
from keras import backend as K 
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, UpSampling2D
from keras.models import Model
from tensorflow.keras.models import load_model

#from keras.utils import multi_gpu_model
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TENSORFLOW_FLAGS"] = "device=gpu%d" % (0)
np.random.seed(1200)  # For reproducibility

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%%
class glVar():
    ""  

# %% Generates data
#Creates two matrix arrays: One in ascending order and
#One array in descending order
#Return the concantenation of those two arrays and another array for labels
def generateTestSequence(m, n):
    # return [randint(0, 4) for _ in range(length)]
    arr = np.empty([0, n])
    for i in range(1, m + 1):
        arr = np.append(arr, [np.arange(i, n + i)], axis=0)
    arr = np.concatenate((arr, np.flip(arr)))
    lab = np.concatenate((np.full((m, 1), 0), np.full((m, 1), 1)))
    # print(arr1.shape)
    return (arr, lab)

# %% Generates data
#Creates two matrix arrays: One in ascending order and
#One array in descending order
#Return the concantenation of those two arrays and another array for labels
def generateTrainSequence(m, n):
    # return [randint(0, 4) for _ in range(length)]
    arr = np.empty([0, n])
    for i in range(1, m + 1):
        arr = np.append(arr, [np.arange(i, n + i)], axis=0)
    #arr = np.concatenate((arr, np.flip(arr)))
    lab = np.full((m, 1), 0)
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
        return "BIN"    
       
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
    #act = "sigmoid"
    def myNet(self, data, act1 = "relu", act2 = "relu", numClasses = 2):
        """"""
        # input = 28 x 28 x 1 (wide and thin)
        hidden1 = Flatten()(data)
        hidden1 = Dense(400, activation= act1)(hidden1)
        hidden1 = Dense(200, activation= act1)(hidden1)
        hidden1 = Dense(100, activation= act1)(hidden1)
        hidden1 = Dense(50, activation= act1)(hidden1)
        hidden1 = Dropout(0.2)(hidden1)
        hidden1 = Dense(numClasses, activation=act2)(hidden1)
        return hidden1
    
    #Code to run Autoencoder
    def runNN(self, X_train, Y_train, X_test, Y_test, X_val, Y_val, 
              Y_train_label= None, Y_test_label = None, 
              Y_val_label = None, act1 = "relu",  act2 = "elu", 
              epochs = 10, batch_size = 128, testAct = False, mod = '', 
              train_model = True, folder_NN_hist = "NN_Hist"):  
      
        weight_file = os.path.join(folder_NN_hist, "BIN_weights_" + str(mod) +".h5").replace(r'\'', '/');
        model_file = os.path.join(folder_NN_hist, "BIN_model_" + str(mod)).replace(r'\'', '/'); 
        hist_file = os.path.join(folder_NN_hist, "BIN_history_" + str(mod) + ".csv").replace(r'\'', '/');
         
        if not testAct:
            if mod == "16qam": act1 = 'relu'; act2 = 'elu'
            elif mod == "8psk": act1 = 'selu'; act2 = 'selu'
            elif mod == "bpsk": act1 = 'selu'; act2 = 'softmax'
            elif mod == "qpsk": act1 = 'elu'; act2 = 'elu'            
            elif mod == "gmsk": act1 = 'relu'; act2 = 'exponential'                
            elif mod == "cpm":act1 = 'elu'; act2 = 'softsign'
            else: act1 = act1; act2 = act2
        if act1 == "": act1 = "relu"
        if act2 == "": act2 =  "softmax"
    
        input_img = Input(shape=(X_train.shape[1], X_train.shape[2], 1))
    
        time_train_start = time.time()
        if train_model:
            #Removes previous files to prevent file creation errors
            if not os.path.exists(folder_NN_hist): os.makedirs(folder_NN_hist)
            if os.path.exists(weight_file): os.remove(weight_file)
            if os.path.exists(model_file): os.remove(model_file)
            if os.path.exists(hist_file): os.remove(hist_file)

            model = Model(input_img, self.myNet(input_img, act1 = act1, act2 = act2, 
                                                   numClasses = Y_test.shape[1]))
            model.compile( optimizer = 'adam', loss='mean_squared_error', 
                             metrics = ['accuracy'])
            model.summary() 
            print("X Train Shape", X_train.shape)
            print("Y Train Shape", Y_train.shape)
            hist = model.fit(X_train, Y_train,
                                          validation_data=(X_val, Y_val), 
                                          batch_size=batch_size,epochs=epochs,
                                          verbose=1, shuffle=False)
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
        score = model.evaluate(X_test, Y_test, verbose=0)
        pred =  (model.predict(X_test, verbose = 0))
        time_test = np.round(time.time() - time_test_start, 2)
        #print("Prediction Shape", pred.shape)
        #print("Data shape: ", x.shape)
        print("Train Data Shape: ", X_train.shape)
        print("Accuracy ", score[1])
        print("Loss: ", score[0], '\n')
        print("Training Time: ", time_train)
        print("Testing Time: ", time_test)
        #Validation loss is taken as the final value in the array of validation loss in the training data
        #Returns Test loss, test accuracy, validation loss, validation accuracy 
        return (score[0], score[1], loss_val_train[-1], acc_val_train[-1], 
                pred, act1, act2, time_train, time_test)  

#%%    
def test():
    NNet = NN()
    NNet.__init__
    x_train, y_train = generateTestSequence(500, 20)
    x_test, y_test =  generateTestSequence(300, 20)
    x_test = NNet.shuffleData(x_test)
    y_test = NNet.shuffleData(y_test)    
    x_test, x_val = np.split(np.asarray(x_test), 2)
    y_test, y_val = np.split(np.asarray(y_test), 2)
    
    x_train = x_train.reshape(-1, 1, x_train.shape[1], 1)/np.max(x_train)
    x_test = x_test.reshape(-1, 1, x_test.shape[1], 1)/np.max(x_test)
    x_val = x_val.reshape(-1, 1, x_val.shape[1], 1)/np.max(x_val)
    print(x_train.shape)
    NNet.runNN(x_train, y_train, x_test, y_test, 
               x_val, y_val, batch_size = 128, epochs = 10,
               act1 = 'relu', act2  = 'elu', train_model = True)
    # #getError(glVar.x_test, glVar.x_pred)
    return 0

#%%
if __name__ == '__main__':
    for i in range (1, 10):
        test()
