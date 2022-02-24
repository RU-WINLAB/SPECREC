# -*- coding: utf-8 -*-
"""
Devoloper:  Tina Burns
School: Rutger University
Advisor: Rchard Martin
Python Version: Python3

This code is used to test a match filter
"""
import numpy as np, scipy, pandas as pd, os, time, scipy
import matplotlib.pyplot as plt, random 
from keras import backend as K 
from scipy import signal
from numpy import genfromtxt
from sklearn.preprocessing import normalize

class glVar():
    temp = None
    temp1 = 0
    myDict = {}
    mag = np.array([])
    pred_mod = np.array([])
    pred_stat = np.array([])
    printData = False
#%%
def generateData(path = "samples", num_points = 1000, samples = 1, posStart = 0):
    if not os.path.exists(path): os.makedirs(path)
    sig_mod  = np.empty([0, 1])
    sig_data  = np.empty([0, num_points])
    for fname in os.listdir(path):
        if fname != ".DS_Store": #Ignores .DS_Store file
            sig = np.fromfile(open(path + '/'+fname), dtype=scipy.float32)[posStart:num_points*samples]
            sig = sig/np.max(sig)
            sig_data = np.append(sig_data, sig.reshape(samples, num_points), axis = 0)
            sig_mod = np.append(sig_mod, np.full(samples, fname))
    return (sig_data, sig_mod)

#%%
def generateData2(path = "samples", num_points = 1000, samples = 1, posStart = 0):
    sig_mod  = np.empty([0, 1])
    sig_data  = np.empty([0, 8*num_points])
    myDict = {
        "bpsk": [0, 0, 1, 1, 0, 0, 1, 1], 
        "qpsk": [0, 0, 0, 0, 1, 1, 1, 1],
        "8psk": [1, 1, 1, 0, 0, 0, 1, 1],
        "16qam": [0, 1, 0, 1, 0, 1, 0, 1]
        }
    for mods in myDict.keys():
        sig_data = np.append(sig_data, [np.tile(myDict[mods], num_points)], axis = 0)
        sig_mod = np.append(sig_mod, mods)
    sig_data = np.repeat(sig_data, samples, axis = 0)
    sig_mod = np.repeat(sig_mod, samples, axis = 0)
    return (sig_data, sig_mod)

#%%
def generateData3(path = "samples", num_points = 1000, samples = 1, posStart = 0):
    samples = samples*4
    if samples == 4: sig_mod = ["bpsk", "qpsk", "8psk", "16qam"]
    else: sig_mod  = random.choices(["bpsk", "qpsk", "8psk", "16qam"], k=samples)
    sig_data  = np.random.random ([samples, num_points]) * 10
    return (sig_data, np.asarray(sig_mod))


# %%
class NN():
    def __init__(self):
        """"""
        # clears preious keras nn session
        K.clear_session()
    
    def getType(self):
        return "MATCH"
        
    def correlateTest(self, sig_filters = "", sig_types = [],  sig_test = "",
                      sig_type = ""):
        #Normalizez filter data.  
        sig_filters = sig_filters/np.max(sig_filters) 
        #Initializes an np array for the complex test signal
        sig_complex = np.full((sig_filters.shape[0], int(sig_test.shape[1]/2)), 0+0j) 
        glVar.pred_mod = np.array([])
        glVar.mag = np.array([])
        glVar.pred_stat = np.array([])
        glVar.mag = []
        for sig, mod in zip(sig_test, sig_type):
            #normalizes signal
            #sig = (sig - np.min(sig))/(np.max(sig)- np.min(sig))
            sig_complex[sig_complex.shape[0]-1][:] = sig[0::2] + 1j*sig[1::2]
            #Convolves the filter with test signal (MATCH FILTER)
            sig_corr = signal.correlate(sig_filters, sig_complex, mode='full')
            #if os.path.exists(mod+'.csv'): os.remove(mod+'.csv')
            # np.savetxt(mod+'.csv',sig_corr, delimiter=',')
            mag = np.max(abs(sig_corr), axis=1)
            #print("MAG: ", mag)
            #argmax returns the location of the array component with the max value
            pred = sig_types[np.argmax(mag)]
            glVar.mag = np.append(glVar.mag, max(mag))
            glVar.pred_mod = np.append(glVar.pred_mod, pred)
            glVar.pred_stat = np.append(glVar.pred_stat, int(pred == mod))
            #plt.plot(sig[0::2], sig[1::2], "*")
            #glVar.temp = sig_corr
        glVar.mag = np.round(np.asarray(glVar.mag), 2)
        return 0

# %% Main NN Execution
    def runNN(self, X_train, Y_train, X_test, Y_test, X_val, Y_val, 
              Y_train_label= None, Y_test_label = None, 
              Y_val_label = None, act1 = "n/a",  act2 = "n/a", 
              epochs = 10, batch_size = 128, testAct = False, mod = '',
              train_model = True, folder_NN_hist ="NN_Hist" ):

        file_IQ = os.path.join(folder_NN_hist, "Match_IQ.csv")
        file_match_info = os.path.join(folder_NN_hist, "Match_Info.csv")
        file_IQ_Complex = os.path.join(folder_NN_hist, "Match_IQ_Complex.csv")
 
        time_train_start = time.time()
        #Training
        if train_model:    
            #Removes previous files to prevent file creation errors
            if not os.path.exists(folder_NN_hist): os.makedirs(folder_NN_hist)
            if os.path.exists(file_IQ): os.remove(file_IQ)
            if os.path.exists(file_IQ_Complex): os.remove(file_IQ_Complex)
            if os.path.exists(file_match_info): os.remove(file_match_info)
            #Writes values to the file_IQ
            with open(file_IQ_Complex, "a") as f:
                for i in X_train:
                    #i= (i - np.min(i))/(np.max(i)- np.min(i))
                    np.savetxt(f, [i[0::2] + 1j*i[1::2]], delimiter=',')
            f.close
        np.savetxt(file_IQ, X_train, delimiter=",")        
        time_train = np.round(time.time() - time_train_start, 2)
        
        #Testing
        time_test_start = time.time()
        X_test = X_test/np.max(X_test)
        #X_val = X_val/np.max(X_val)
        self.correlateTest(
            sig_filters = genfromtxt(file_IQ_Complex, dtype=complex, delimiter=','), 
            sig_types = Y_train_label,
            sig_test = X_test, sig_type = Y_test_label)
        
        #Gets and outputs predciton of each class
        pred = glVar.pred_mod
        acc_val = 0
        acc_test = np.sum(glVar.pred_stat)/glVar.pred_stat.shape[0]
        time_test = np.round(time.time() - time_test_start, 2)
        #print("Data shape: ", x.shape)
        #print("Train Data Shape: ", X_train.shape)
        print("Accuracy ", acc_test)
        print("Train Shape: ", X_train.shape)
        print("Test Shape: ", X_test.shape)
        #print(glVar.pred_mod)
        #print(Y_test_label)
        
        #if glVar.printData: """""""

        #Returns Test loss, test accuracy, validation loss, validation accuracy 
        return (-1, acc_test, -1, acc_val, Y_val_label, 'avg', '',
                time_train, time_test)  

#%%
def subplot_data(arr):
    #https://matplotlib.org/tutorials/introductory/sample_plots.html#sphx-glr-tutorials-introductory-sample-plots-py
    #import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(5, 5))
    
    axs[0, 0].plot([x.real for x in arr[0]], [y.imag for y in arr[0]])
    axs[0, 0].title.set_text(glVar.filter_sig_type[0])
    axs[1, 0].plot([x.real for x in arr[1]], [y.imag for y in arr[1]])
    axs[1, 0].title.set_text(glVar.filter_sig_type[1])
    axs[0, 1].plot([x.real for x in arr[2]], [y.imag for y in arr[2]])
    axs[0, 1].title.set_text(glVar.filter_sig_type[2])
    axs[1, 1].plot([x.real for x in arr[3]], [y.imag for y in arr[3]])
    axs[1, 1].title.set_text(glVar.filter_sig_type[3])
    plt.show()
    return 0

# %%
def test():
    #print (sig[0:5])
    NNet = NN()
    NNet.__init__

    X_train, Y_train_label = generateData(path = 'Data/mod_test2/snr30', num_points = 250, samples = 1)              
    X_test, Y_test_label = generateData(path = 'Data/mod_test2/snr20', num_points = 1000, samples = 10)
    #X_train = normalize(X_train, axis = 1, norm = 'l1')
    #X_test = normalize(X_train, axis = 1, norm = 'l1')
    #X_test = X_test.reshape(-1, 1, X_test.shape[1], 1)/np.max(X_test)
    NNet.runNN(X_train=X_train, Y_train_label=Y_train_label, X_test=X_test, 
                Y_test_label=Y_test_label, X_val=0, Y_val=0, Y_train = 0, Y_test = 0)
    #subplot_data((glVar.temp))
    glVar.temp = Y_test_label
    return 0

if __name__ ==  '__main__':
    #glVar.printData = True
    y = test()

