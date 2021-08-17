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
import os, keras, numpy as np 
import pandas as pd
from keras.layers import Dense, Dropout, Input
from keras import backend as K 
from keras.models import Model
from keras.callbacks import TensorBoard, History
history = History()

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#from keras.utils import multi_gpu_model
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TENSORFLOW_FLAGS"] = "device=gpu%d" % (0)
np.random.seed(1200)  # For reproducibility
from tensorflow.keras.models import load_model
# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%%
class glVar():
    x_test = []
    y_test = []
    y_test_label = []
    x_val = []
    y_val = []
    y_val_label = []
    x_train = []
    y_train = []
    y_train_label = []
    x_pred = []
    y_pred = []
    err = []
    diff = []
    prec = []
    recall = []
    thres1 = []
    err = []
    mse = []
    mse_train = []
    thres = 0
    acc_test = 0
    acc_val = 0
    error_df = pd.DataFrame()
    
    

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
        return "ANOM"
        
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
  
    
    #Code to run Autoencoder
    def runNN(self, X_train, Y_train, X_test, Y_test, 
              X_val, Y_val, Y_train_label= None, Y_test_label = None, 
              Y_val_label = None, batch_size=128, epochs=10, 
              act1 = 'relu', act2 = 'softmax', testAct = False, mod = '', 
              NN_layers = "CONV", plotCurve = False, train_model = True,
              folder_NN_hist = "NN_Hist"):

        weight_file = os.path.join(folder_NN_hist, "ANOM_weights.h5"); 
        model_file = os.path.join(folder_NN_hist, "ANOM_model"); 
        hist_file = os.path.join(folder_NN_hist, "ANOM_history.csv");
         
        if not testAct:
            if mod == "16qam": act1 = 'selu'; act2 = 'softsign'
            elif mod == "8psk": act1 = 'selu'; act2 = 'softsign'
            elif mod == "bpsk": act1 = 'selu'; act2 = 'elu'                
            elif mod == "qpsk": act1 = 'selu'; act2 = 'relu'
            elif mod == "cpm": act1 = 'relu'; act2 = 'tanh'
            elif mod == "gmsk": act1 = 'hard_sigmoid'; act2 = 'softmax'
            else: act1 = act1; act2 = act2
        if act1 == "": act1 = "relu"
        if act2 == "": act2 =  "softmax"
        
        # Autoencoder model fit and compile
        # decoder(encoder(input_img))
        input_img = Input(shape=(X_train.shape[1], X_train.shape[2], 1))
        
        if train_model:
            if not os.path.exists(folder_NN_hist): os.makedirs(folder_NN_hist)
            model_AE = Model(input_img, self.decoder(self.encoder(input_img, act1 = act1), 
                                                     act1 = act1, act2 = act2))
            model_AE.compile( optimizer = 'adam', loss='binary_crossentropy', 
                             metrics = ['accuracy'])
            
            print("Storing Weights... \n")
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, 
                                                       verbose=0, mode='auto')
            check_point = keras.callbacks.ModelCheckpoint(weight_file, monitor='val_loss', verbose=0,
                                                          save_best_only=True, mode='auto')
            validation_info = keras.callbacks.Callback()
            tb = TensorBoard(log_dir='./logs',
                    histogram_freq=0,
                    write_graph=True,
                    write_images=True)
            print("Summary")
            model_AE.summary() 
            model_AE.save(model_file)
            #model_AE.load_weights(weight_file)
            print("Training Data...")
            hist = model_AE.fit(X_train, Y_train,
                                          validation_data=(X_val, Y_val), 
                                          batch_size=batch_size,epochs=epochs,
                                          callbacks=[check_point, early_stop, tb, history],
                                          verbose=1, shuffle=False)
        
            loss_test_train = np.asarray(hist.history['loss'])
            acc_test_train = np.asarray(hist.history['accuracy'])
            loss_val_train = np.asarray(hist.history['val_loss'])
            acc_val_train = np.asarray(hist.history['val_accuracy'])
            
            pd.DataFrame({"loss_val_train" : loss_val_train,
                       "acc_val_train" : acc_val_train,
                        }).to_csv(hist_file)
        else:
            model_AE = load_model(model_file)
            #model()
            hist = pd.read_csv(hist_file)
            #print(hist)
            loss_val_train = hist["loss_val_train"].values
            acc_val_train = hist["acc_val_train"].values                    
        
        score = model_AE.evaluate(X_test, Y_test)
        pred_net = model_AE.predict(X_train, verbose = 1)
        glVar.temp = score

        glVar.mse_train = np.sum(np.mean(np.power(X_train - pred_net, 2), axis=1), axis =1)
        glVar.mse_train = np.round(np.asarray(glVar.mse_train.reshape(glVar.mse_train.shape[0])), 2)
        
        def accPred(x, y, plotCurve):
            glVar.x_pred =  (model_AE.predict(x))
            #glVar.mse = np.sum(np.mean(np.power(x - glVar.x_pred, 2), axis=1), axis =1)
            glVar.mse = np.sum(np.mean(np.power(x - glVar.x_pred, 2), axis=1), axis =1)
            glVar.thres1 = np.max(glVar.mse_train)/np.max(glVar.mse)
            glVar.mse = np.round(np.asarray(glVar.mse.reshape(glVar.mse.shape[0]))/np.max(glVar.mse), 2)
            
            #If nan values exist in glVar.mse, the function return 0 for the accuracy
            if np.sum(np.isnan(glVar.mse)) > 0: glVar.acc = 0
            else:   
                glVar.y_test_label = np.asarray((y.reshape(y.shape[0])))        
                thres, glVar.y_pred = getPred(glVar.y_test_label, glVar.mse, plotCurve = plotCurve)
                glVar.y_pred = np.asarray(glVar.y_pred)
                glVar.acc = np.sum(np.power(glVar.y_test_label - glVar.y_pred, 2))/glVar.y_pred.shape[0]       
            return glVar.acc

        
        glVar.acc_val = accPred(X_val, Y_val_label, False)
        glVar.acc_test = accPred(X_test, Y_test_label, plotCurve)
        #getPred2(glVar.y_test_label, glVar.mse, plotCurve = False)

        if plotCurve:
            # list all data in history
            print(hist.history.keys())
            # summarize history for accuracy
            fig1, ax1 = plt.subplots()
            ax1.plot(acc_test_train)
            ax1.plot(acc_val_train)
            ax1.set_title('model accuracy')
            ax1.set_ylabel('accuracy')
            ax1.set_xlabel('epoch')
            ax1.legend(['train', 'test'], loc='upper left')
            #ax1.show()
            # summarize history for loss
            fig2, ax2 = plt.subplots()
            ax2.plot(loss_test_train)
            ax2.plot(loss_val_train)
            ax2.set_title('model loss')
            ax2.set_ylabel('loss')
            ax2.set_xlabel('epoch')
            ax2.legend(['train', 'test'], loc='upper left')
            #ax2.show()
    
        print("Threshold: ", glVar.thres)
        print("Test Accuracy: ", glVar.acc_test)
        print("Test Loss: ", score[0])
        #print("Validation Accuracy: ",  glVar.acc_val)        
        #print("Validation Loss: ",  loss_val_train[-1])
        
        #Validation loss is taken as the final value in the array of validation loss in the training data
        #Returns Test loss, test accuracy, validation loss, validation accuracy 
        return score[0], glVar.acc_test, loss_val_train[-1], glVar.acc_val, glVar.y_pred, act1, act2  
        # print(hist.history)
        #return hist
    

#%%
#part of code from
#https://towardsdatascience.com/extreme-rare-event-classification-using-autoencoders-in-keras-a565b386f098
def getPred(y_lab, err, plotCurve = False):
    glVar.prec, glVar.recall, glVar.thres1 = precision_recall_curve(y_lab, err)
    glVar.prec = glVar.prec[1:] 
    glVar.recall = glVar.recall[1:]
    #Removes precions values equal to 0 to prevent anomalies
    prec0 = np.where(glVar.prec==0)
    glVar.prec = np.delete(glVar.prec, prec0)
    glVar.recall = np.delete(glVar.recall, prec0)
    glVar.thres1 = np.delete(glVar.thres1, prec0)
    
    if plotCurve:
        plt.plot(glVar.thres1, glVar.recall, label="Recall",linewidth=5)
        plt.plot(glVar.thres1, glVar.prec, label="Precision",linewidth=5)
        plt.title('Precision and recall for different threshold values')
        plt.xlabel('Threshold')
        plt.ylabel('Precision/Recall')
        plt.legend()
        plt.show()
    
    glVar.diff =  np.power(glVar.prec - glVar.recall, 2)
    
    loc = (np.where(glVar.diff == np.min(glVar.diff)))
    myThres = glVar.thres1[loc]
    #print ("Estimated Threshold: ", myThres)
    #print("Location: ",i)
    #print("MSE Precision and Recall: `\n", diff)
    
    glVar.thres = myThres[0]
    pred = []
    for i in err:
        if i > glVar.thres: pred.append(0);
        else: pred.append(1)
    return glVar.thres, pred
    return 0, 1


#%%
#Gets prediction values based on the max MSE of the training data
def getPred1(y_lab, err, plotCurve = False):

    glVar.thres = glVar.thres1
    #glVar.thres = max(glVar.mse_train)/max(glVar.mse)
    pred = []
    for i in err:
        if i > glVar.thres: pred.append(0);
        else: pred.append(1)
    return glVar.thres, pred

#%%
#Gets prediction values based on the max MSE of the training data
def getPred2(y_lab, err, plotCurve = False):
    fpr, tpr, thresholds = roc_curve(y_lab, err)
    #glVar.thres = glVar.thres1
    #glVar.thres = max(glVar.mse_train)/max(glVar.mse)
    
    if plotCurve:
        plt.plot(fpr,tpr) 
        #plt.axis([0,1,0,1]) 
        plt.xlabel('False Positive Rate') 
        plt.ylabel('True Positive Rate') 
        plt.title("ROC Curve")
        plt.show()   
    
    
    pred = []
    for i in err:
        if i > glVar.thres: pred.append(0);
        else: pred.append(1)
    return glVar.thres, pred

#%%    
def test():
    NNet = NN()
    NNet.__init__
    glVar.x_train, glVar.y_train_label = generateTrainSequence(300, 100)
    glVar.x_test, glVar.y_test_label =  generateTestSequence(100, 100)
    glVar.x_test = NNet.shuffleData(glVar.x_test)
    glVar.y_test = NNet.shuffleData(glVar.y_test_label)    
    glVar.x_test, glVar.x_val = np.split(np.asarray(glVar.x_test), 2)
    glVar.y_test_label, glVar.y_val_label = np.split(np.asarray(glVar.y_test), 2)
    
    
    glVar.x_train = glVar.x_train.reshape(-1, 1, glVar.x_train.shape[1], 1)/np.max(glVar.x_train)
    glVar.x_test = glVar.x_test.reshape(-1, 1, glVar.x_test.shape[1], 1)/np.max(glVar.x_test)
    glVar.x_val = glVar.x_val.reshape(-1, 1, glVar.x_val.shape[1], 1)/np.max(glVar.x_val)
    
    
    
    print(glVar.x_train.shape)
    NNet.runNN(glVar.x_train, glVar.x_train, glVar.x_test, glVar.x_test, 
               glVar.x_val, glVar.x_val, batch_size = 128, epochs = 10,
               act1 = 'exponential', act2  = 'elu', 
               Y_test_label = glVar.y_test_label, 
               Y_val_label = glVar.y_val_label, 
               train_model = True)
    #getError(glVar.x_test, glVar.x_pred)
    return 0
#%%
if __name__ == '__main__':
   test()
