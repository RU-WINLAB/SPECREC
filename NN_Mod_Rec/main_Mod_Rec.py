#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Devoloper:  Tina Burns
School: Rutger University
Advisor: Rchard Martin
Python Version: Python3

This code sequeces to through neural network for modulation recognition
"""
#Imports necessary libraries 
import numpy as np, os, glob
from datetime import datetime
import sys, ntpath, time
import pandas as pd
from argparse import ArgumentParser
#import matplotlib.pyplot as plt
from keras import backend as K 
np.random.seed(1200)  # For reproducibility

#Allows use of modules from the Common_Functions Folders
sys.path.append('../../_Neural_Networks')

#FILE IMPORTS
# Imports standard NN files
import NN_FCNN_b14 as NN_CAT
import NN_CNN_b19 as NN_CAT_CONV
import NN_BIN_b14 as NN_BIN
import NN_AE_b14 as NN_AE
import NN_ANOM_b15 as NN_ANOM
import NN_LSTM_b7 as NN_LSTM
import NN_LSTM_64L as NN_LSTM_2
import NN_Simple_b3 as NN_SIMPLE
import NN_matched_filter_b13 as MATCH
# Imports "normalized"NN files
import NN_FCNN_norm_b1 as NN_CAT_2
import NN_CNN_norm_b1 as NN_CAT_CONV_2
import NN_BIN_norm_b1 as NN_BIN_2
import NN_AE_norm_b1 as NN_AE_2
# Imports additional files
import compare_prediction_actual_r4 as conf_mat
import read_binary_r1 as read_binary_file
import warnings                                                                                                                
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)


#%%
#This class allows global variable to be access by all functions
class glVar ():
    IQ_pair = np.array([[], []])
    fileNames = np.array([])
    mod_type = []
    mod_list = []
    mod_int = []
    mod_UT = ""
    snr = []
    train_x =  np.array([])
    train_y = np.array([])
    train_label = np.array([])
    test_x = np.array([])
    test_y = np.array([])
    test_label = np.array([])
    val_x = np.array([])
    val_y = np.array([])
    val_label = np.array([])

    perm = np.array([])
    data_hex = []
    sep_train_test = True

    myFile = ""
    logfile = []
    folder_base = "C:/Users/TINA/OneDrive - Rutgers University/Rutgers/Research/SDR/Data/"
    
    dtype = "float32"
    folder_test = ""
    folder_train = ""
    folder_results = ""
    myResults = pd.Series([])
    #featureData = pd.Series([])
    testData = pd.Series([])
    test_data = {}
    train_data = {}
    train_X_size = 0
    filePos = 0
    dataArr = {}
    NN_type = ""
    dateCode = ""
    num_points_train = 250
    NNets = ["CAT", "CAT_CONV", "AE",  "BIN","ANOM"]
    header = True
    temp = None
    temp1 = None
    pred = []
    param_value = []
    col_param = ""
    col_mods = "s1_mod"
    cycle = 0
    time_start_OVH = 0
    time_data_collect = 0
    
    NN_train = 1
    NN_Hist_folder = "NN_Hist"
    exc_type = []
    exc_list_train = []
    exc_list_test = []
    iter_f = 0; 
    iter_test_dat = 0
                                
#%%
def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--folder-train", dest="folder_train", type=str, 
        default= "C:/Users/TINA/OneDrive - Rutgers University/Rutgers/Research/SDR/Data/2020-06-10_additive-noise_gNoise-2",
        help="Sets the training data folder location [default=%(default)r]")
    parser.add_argument(
        "--folder-test", dest="folder_test", type=str, 
        default= "",
        help="Set testing data folder location [default=%(default)r]")
    parser.add_argument(
        "--neural-nets", dest="NNets", type=str, nargs='+', 
        default= ["CAT", "CONV", "LSTM", "BIN", "ANOM",  "AE"],
        help='''Enter the list of nueral networks to be tested. \n
        BIN --> Binary Classifier \n
        FCN --> Fully Connected \n
        CNN --> Convolutional \n
        AE --> Autoencoder \n
        LSTM --> LSTM with 32 filters  \n
        AMOM --> Anomaly Detector \n
        SIMP--> Simple \n
        BIN --> Binary Classifier \n
        FCN2 --> Normalized Fully Connected  \n
        CNN2 --> Normalized Convolutional \n
        AE2 --> Normalized Autoencoder \n
        LSTM2 --> LSTM with 64 filters  \n
        Options [default=%(default)r]'''
        )
    parser.add_argument(
        "--iter", dest="iter", type=int, 
        default= "1",
        help="Number of iterations [default=%(default)r]")
    parser.add_argument(
        "--samples", dest="samples", type=int, 
        default= "10000",
        help="Number of samples [default=%(default)r]")
    parser.add_argument(
        "--num-points", dest="num_points", type=int, 
        default= 1000,
        help="Number of datapoints [default=%(default)r]")
    parser.add_argument(
        "--num-points-train", dest="num_points_train", type=int, 
        default= "250",
        help="Number of samples [default=%(default)r]")
    parser.add_argument(
        "--test-act", dest="test_act", type=str, 
        default= "0",
        help="Do you want to test activation functions? Enter 0 for no and 1 for yes. [default=%(default)r]")    
    parser.add_argument(
        "--conf-mat", dest="conf_mat", type=int, 
        default= 0,
        help="Do you want to plot confusion matrices? Enter 0 for no and 1 for yes. [default=%(default)r]")    

    parser.add_argument(
        "--exc-param", dest="exc_param", type=str, 
        default= "s1_mod",
        help="Name of the parameter to be exluded [default=%(default)r]")
    parser.add_argument(
        "--exc-train", dest="exc_train", type=str, nargs='+',
        default= [""],
        help="Values in training data to be exluded [default=%(default)r]")
    parser.add_argument(
        "--exc-test", dest="exc_test", type=str, nargs='+',
        default= [""],
        help="Values in test data to be exluded [default=%(default)r]")
    parser.add_argument(
        "--range-param", dest="range_param", type=str, 
        default= "s1_sinr",
        help="Name of the parameter to be evaulated based on range [default=%(default)r]")
    parser.add_argument(
        "--range-train", dest="range_train", type=float, nargs='+',
        default= [-1000.0, 1000.0],
        help="Range of values in training data to be included [default=%(default)r]")
    parser.add_argument(
        "--range-test", dest="range_test", type=float, nargs='+',
        default= [-1000.0, 1000.0],
        help="Range of values in test data to be included [default=%(default)r]")
    
    parser.add_argument(
        "--col-param", dest="col_param", type=str, 
        default= "s1_sinr",
        help="Name of column in logfile for parameter to be tested [default=%(default)r]")  
    # parser.add_argument(
    #     "--col-filename", dest="col_filename", type=str, 
    #     default= "filename",
    #     help="Name of column in logfile for parameter for filenames [default=%(default)r]")        
    parser.add_argument(
        "--col-mods", dest="col_mods", type=str, 
        default= "s1_mod",
        help="Enter the name of column header for modulations [default=%(default)r]")
    parser.add_argument(
        "--logfile", dest="logfile", type=str, nargs='+', 
        default= [""],
        help="List of logfiles [default=%(default)r]")
    parser.add_argument(
        "--NN-train", dest="NN_train", type=int, 
        default= 1,
        help="Do you want to train the neural network? Enter 0 for no and 1 for yes. [default=%(default)r]")   
    parser.add_argument(
        "--NN-Hist-folder", dest="NN_Hist_folder", type=str, default= "NN_Hist",
        help="Sets the training data folder location [default=%(default)r]")
    parser.add_argument(
        "--data-type", dest="data_type", type=str, 
        default= "float32",
        help='''Sets the data type for reading the file  [default=%(default)r]
        int8 --> 8 bit signed integar '\n'
        int16 --> 16 bit signed integar '\n'
        int32 --> 32 bit signed integar '\n'
        int64 --> 64 bit signed integar '\n'
        uint8 --> 8 bit unsigned integar '\n'
        uint16 --> 16 bit unsigned integar '\n'
        uint32 --> 32 bit unsigned integar '\n'
        uint64 --> 64 bit unsigned integar '\n'        
        float32 --> 32 bit floating point number '\n'
        float64 --> 64 bit floating point number '\n'
        ''')    
    return parser

#%%
#This portion of the code was written by Ryan Davis and modified by Tina Burns. 
def getFileData(data_path, num_points_per_sample, num_samples_per_file, 
                posStart = 0, testing = True, arr_exc = []):
    
    x = []; y = []; z = []; count = 0;  glVar.param_value = []
    for fname in os.listdir(data_path):
        #Ignores .DS_Store file ang files that don't meet the parameter specifications
        if (fname != ".DS_Store" and fname not in arr_exc and fname.find(".txt")<0 and fname.find(".csv")<0): 
            f = read_binary_file.read_binary_iq(fname = data_path+ '/'+fname, samples = num_samples_per_file, 
                    num_points = num_points_per_sample, pos_sample = posStart, d_type = glVar.dtype)
            #f = (np.asarray(f)/np.max(abs(np.asarray(f))))
            f = f[:][0:(f.shape[0] - f.shape[0]%num_points_per_sample)] #Ensures even number of points
            #print(posStart)
            if count == 0: x = f.reshape(-1, num_points_per_sample)[0:num_samples_per_file]
            else: x = np.vstack((x, f.reshape(-1, num_points_per_sample)[0:num_samples_per_file]))
            y = y + [ntpath.basename(fname)]*num_samples_per_file
        
            #The portion of the fuction get the modulation information
            mod = glVar.testData[glVar.testData["filename"] == fname][glVar.col_mods].values.item()
            z = z + [mod]*num_samples_per_file            
            if testing: 
                glVar.param_value.append(float(glVar.testData[glVar.testData["filename"] == 
                    fname][glVar.col_param].values.item()))               
                count = count +1
    return x, np.asarray(y), np.asarray(z)
#%%
# Shuffles Data
def shuffleData(x):
    np.random.seed(1200)
    myPermutation = np.random.permutation(x.shape[0])
    x = x[myPermutation]
    return x

#%%
def getExclusionList(range_param  = "s1_sinr", range_arr = [-1000, 1000], exc_param = "s1_mod", exc_arr = [""]):
    arr = []
    #Makes all values in exculsion column lower case
    glVar.testData[exc_param] = glVar.testData[exc_param].str.lower() 
    for exc in exc_arr:
        arr = arr + list(glVar.testData["filename"][glVar.testData[exc_param] == exc.lower()])
    #print([glVar.testData[range_param] >= range_arr[0]])
    arr = arr + list(glVar.testData["filename"][glVar.testData[range_param] < range_arr[0]])
    arr = arr + list(glVar.testData["filename"][glVar.testData[range_param] > range_arr[1]])
    #print("making exclusion list")
    return arr
#%%
#Main function of the program tha executes the main operations
def genData(myFile, numDatapoints = 100, numSamples = 200, pos = 0, 
            mod = "", NN_Type = "CAT", testing = True, arr_exc = []):    
    my_dict = {}
    #Inputs information into global variables for later usage
    #The number of bytes must be divisible by 8 in order to properly work with the NN
    glVar.IQ_pair, glVar.fileNames, glVar.mod_type = getFileData(myFile, numDatapoints, numSamples, 
            posStart = pos, testing = testing, arr_exc = arr_exc)      
    
    if len(glVar.IQ_pair) >= 1:
        #Shuffles the all data arrays
        glVar.IQ_pair = shuffleData(glVar.IQ_pair)
        glVar.fileNames = shuffleData(glVar.fileNames)
        glVar.mod_type = shuffleData(glVar.mod_type)
        
        #Get unique values of modulation schemes
        glVar.mod_list = pd.get_dummies(glVar.mod_type)
        #Puts mod_int list in the form of integar values
        if glVar.NN_type  == "ANOM":
            glVar.mod_int = pd.factorize(glVar.mod_type)[0]
        #Puts mod_int list in the form of binary arrays (for categorical classification)
        else:
            glVar.mod_int = glVar.mod_list.values
        #Stores information in dictionary
        my_dict = {
           #"IQ_stack": glVar.IQ_stack, 
            "IQ_pair": glVar.IQ_pair,
            "mod_type": glVar.mod_type,
            "mod_int": glVar.mod_int,
            glVar.col_param: glVar.param_value
            }
    return my_dict
    
# %%
def runTest(dateCode, datapoints = 100, samples = 200, writeData = True, 
            num_iter = 2, act1 = "", act2 = "", testAct = False, options = None):
    glVar.cycle = glVar.cycle + 1
    #time_start_OVH = time.time()
    epochs = [10]
    dataArr = ["IQ_pair"]
    for NNet_test in glVar.NNets:
        if NNet_test == "ANOM": NNet = NN_ANOM.NN()
        elif NNet_test.upper() == "FCN": NNet = NN_CAT.NN()
        elif NNet_test.upper() == "CNN": NNet = NN_CAT_CONV.NN()
        elif NNet_test.upper() == "BIN": NNet = NN_BIN.NN()
        elif NNet_test.upper() == "AE": NNet = NN_AE.NN()
        elif NNet_test.upper() == "LSTM": NNet = NN_LSTM.NN()
        elif NNet_test.upper() == "SIMPLE": NNet = NN_SIMPLE.NN()
        elif NNet_test.upper() == "MATCH": NNet = MATCH.NN()
        elif NNet_test.upper() == "FCN2": NNet = NN_CAT_2.NN()
        elif NNet_test.upper() == "CNN2": NNet = NN_CAT_CONV_2.NN()
        elif NNet_test.upper() == "BIN2": NNet = NN_BIN_2.NN()
        elif NNet_test.upper() == "AE2": NNet = NN_AE_2.NN()
        elif NNet_test.upper() == "LSTM2": NNet = NN_LSTM_2.NN()
        else: NNet = NN_CAT.NN()
        
        glVar.NN_type = NNet.getType()    
        NNet.__init__
    
        if glVar.NN_type == "BIN" or glVar.NN_type == "ANOM" :
            #Gets list unique list of modulation types
            #modulations = ["bpsk", "qpsk", "16qam", "8psk"]
            modulations = set(glVar.testData[glVar.col_mods].values)
        else:
            modulations = ["all"]
        #modulations = ["bpsk"]

        for i in dataArr:
            """"""
            K.clear_session()
            for j in range(1, num_iter+1):
                for m in modulations:
                    #gets test Data
                    glVar.mod_UT = m
                
                    if glVar.iter_test_dat == 0: print("Generating Training Data")                    
                    #Only generates training data if it is in a different folder
                    if glVar.NN_train == 1:
                        #print("Collecting traning data")
                        train_samples = samples
                        if NNet_test == "MATCH": 
                            train_samples = 1
                            dp = glVar.num_points_train
                        else: dp = datapoints
                        #print("Number of training samples: ", train_samples)
                        glVar.train_data = genData(glVar.folder_train, dp, train_samples, 
                            mod = m, NN_Type = NNet_test, arr_exc = glVar.exc_list_train)
                        glVar.train_y = np.asarray(glVar.mod_int)               
                        glVar.train_label = glVar.mod_type
                        train_model = True
                        glVar.train_x = glVar.train_data[i]
                    else: 
                        #print("Not training model")  
                        train_model = False
                        #print(glVar.train_x.shape)
                        if glVar.train_x.shape[0] <= 1:
                            glVar.train_x = np.zeros((samples, datapoints))     
                            glVar.train_y = np.zeros((samples, 1))                
                            glVar.train_label = np.zeros((samples, 1)) 

                    
                    if glVar.iter_test_dat == 0: print("'\n'Generating Test Data")
                    glVar.iter_test_dat = glVar.iter_test_dat +1
                    glVar.test_data = genData(glVar.folder_test, datapoints, samples//2,
                        pos = datapoints*samples, mod = m, NN_Type = NNet_test, 
                        arr_exc = glVar.exc_list_test)
                    if len(glVar.IQ_pair) < 1: 
                        # Allows for data to be trained if test data is empty and the 
                        # data has not been trained yet                         
                        if glVar.iter_f == 1: glVar.iter_f = 0 
                        continue #Breaks the loop if the IQ array is empty 

                    glVar.test_x, glVar.val_x = np.split(glVar.test_data[i], 2)
                    glVar.test_y, glVar.val_y = np.split(np.asarray(glVar.mod_int), 2)
                    glVar.test_label, glVar.val_label = np.split(glVar.mod_type, 2)
                    print("Test Shape", glVar.test_x.shape)
                    
                    # #This shapes the array so that it is evenly divisible by 4
                    # #which allows it to be correctly processed by the nueral network
                    glVar.train_x = glVar.train_x[:, 0:(glVar.train_x.shape[1] - glVar.train_x.shape[1]%4)]
                    glVar.test_x = glVar.test_x[:, 0:(glVar.test_x.shape[1] - glVar.test_x.shape[1]%4)]
                    glVar.val_x = glVar.val_x[:, 0:(glVar.val_x.shape[1] - glVar.val_x.shape[1]%4)]
                    # Reshapes the data so that it is 4 dimensional
                    # Seperates test and data intoM training, test, and validiation sets
                    #Normalizes x data                   
                    if NNet_test.find("LSTM") > -1:
                        if len(glVar.train_x.shape) <= 2: glVar.train_x = glVar.train_x.reshape(-1, glVar.train_x.shape[1], 1)/np.max(glVar.train_x)
                        glVar.test_x = glVar.test_x.reshape(-1, glVar.test_x.shape[1], 1)/np.max(glVar.test_x)
                        glVar.val_x = glVar.val_x.reshape(-1, glVar.val_x.shape[1], 1)/np.max(glVar.val_x)
                    elif NNet_test != "MATCH":
                        if len(glVar.train_x.shape) <= 2: glVar.train_x = glVar.train_x.reshape(-1, 1, glVar.train_x.shape[1], 1)/np.max(glVar.train_x)
                        glVar.test_x = glVar.test_x.reshape(-1, 1, glVar.test_x.shape[1], 1)/np.max(glVar.test_x)
                        glVar.val_x = glVar.val_x.reshape(-1, 1, glVar.val_x.shape[1], 1)/np.max(glVar.val_x)

                    if not testAct: activations = [""];
                    else: activations = ["elu", "softmax", "selu", "softplus", "softsign", 
                        "relu", "tanh", "sigmoid", "hard_sigmoid", "exponential", 
                        "linear"]; 

                    for a1 in  activations:
                        for a2 in activations:                     
                            for e in epochs:                        
                                if glVar.NN_type == "ANOM":
                                    glVar.train_y = glVar.train_x
                                    glVar.test_y = glVar.test_x
                                    glVar.val_y = glVar.val_x
                                print("Starting Neural Network...")
                                time_NN = time.time()
                                (loss_test, acc_test, loss_val, acc_val, glVar.pred, 
                                  af1, af2, time_train, time_test) = NNet.runNN(
                                    X_train = glVar.train_x, 
                                    Y_train = glVar.train_y,
                                    Y_train_label = glVar.train_label,
                                    X_test = glVar.test_x,
                                    Y_test = glVar.test_y, 
                                    Y_test_label = glVar.test_label,
                                    X_val = glVar.val_x, 
                                    Y_val = glVar.val_y, 
                                    Y_val_label = glVar.val_label,
                                    batch_size = 128, epochs = e, 
                                    mod = m, act1 = a1, act2 = a2,
                                    testAct = testAct, 
                                    #plotCurve = False,
                                    train_model= train_model, 
                                    folder_NN_hist = glVar.NN_Hist_folder
                                    )
                                time_NN = np.round(time.time() - time_NN, 2)                                     
                                atten = 0; snr = 100;
                                name = os.path.basename(glVar.folder_train)
                                if name.find("atten") > -1: atten = name.split("atten")[1]
                                if name.find("snr") > -1:  snr = name.split("snr")[1]
                                time_OVH = np.round(time.time() -glVar.time_start_OVH - time_NN, 2)
                             
                                if writeData:
                                    pd.DataFrame({
                                            #"Datatype" : [i], 
                                            glVar.col_param: [np.round(np.mean(glVar.param_value), 2)],
                                            "Acc-Test": [np.round(acc_test, 3)], 
                                            "Loss-Test ": [np.round(loss_test, 2)],
                                            "Acc-Val": [np.round(acc_val, 3)],        
                                            "Loss-Val ": [np.round(loss_val, 2)], 
                                            "Epochs": [e],
                                            "Train Samples": [glVar.train_x.shape[0]],
                                            "Test Samples": [glVar.test_x.shape[0]],
                                            "Validation Samples": [glVar.val_x.shape[0]],
                                            "Datapoints": [datapoints],              
                                            "dir_train": [os.path.basename(os.path.dirname(glVar.folder_train))],
                                            "folder_train": [os.path.basename(glVar.folder_train)],   
                                            "dir_test": [os.path.basename(os.path.dirname(glVar.folder_test))],
                                            "folder_test": [os.path.basename(glVar.folder_test)],
                                            "NN_Type": [glVar.NN_type],
                                            "Mod-Type": [m], 
                                            "time_data_collect": [glVar.time_data_collect], 
                                            "time_data_manip": [time_OVH], 
                                            "time_NN": [time_NN], 
                                            "time_train": [time_train], 
                                            "time_test": [time_test],
                                            "Activation 1: ": [a1],
                                            "Activation 2": [a2],
                                            "Param ": [options.range_param],
                                            "Param Train Min": [options.range_train[0]],
                                            "Param Train Max": [options.range_train[1]],
                                            "Param Test Min": [options.range_test[0]],
                                            "Param Test Max": [options.range_test[1]],
                                            }).to_csv("Data/Results/" + glVar.dateCode + "_Test.csv", mode = 'a', 
                                                      header = glVar.header)
                                    glVar.header = False
                                    glVar.time_data_collect = 0
                                    
                                print("NN: "+ glVar.NN_type + "  ATTEN: " + str(atten) + "  Mod: " + m)
                                print(" Test Folder: " + os.path.basename(glVar.folder_test)) 
                                print("Time of NN: ", time_NN)
                                print(glVar.col_param, np.round(np.mean(glVar.param_value), 2))
                                print( " Activation 1:  " + af1 + " Activation 2:  " + af2)
                                if options.conf_mat: 
                                    #Sets up prediction array as a categorical array
                                    glVar.temp1 = glVar.pred
                                    if glVar.NN_type == "MATCH": 
                                        labels = pd.get_dummies(glVar.pred).columns.tolist()
                                        glVar.pred =  np.asarray(pd.get_dummies(glVar.pred).values)
                                    else: labels = list(glVar.mod_list.columns.values)
                                    conf_mat.main(y_pred = glVar.pred, y_true = glVar.test_y,
                                        #Gets a list of modulation types that aligns with binary array 
                                        labels_text = labels, 
                                        myFolder = "Data/Results/", 
                                        myFile =glVar.dateCode + "_" + os.path.basename(glVar.folder_test) + "_"
                                        + glVar.NN_type)
                                    glVar.temp = labels
 
#%% Gets list of folders to be tested 
def getFolderList(loc_data):
    #a = [*set([loc_data + "/" + p for p in os.listdir(loc_data)])]
    a = []; 
    #os.join create adds a '\' when joining info.  
    for root, dirs, files in os.walk(loc_data):
        for d in dirs:
            if(d.lower().find("data") <0 and d.lower().find("plot") <0 and d.find("cleansig")<0): 
                a.append( str(root + '/' + d).replace("//", "/"))
        for f in files:
            if (f.lower().find("log") > -1 or f.lower().find(".csv") > -1): 
                glVar.logfile.append(root + "/" + f)
                
    if len(a) == 0: a.append(loc_data)
    if not os.path.exists("Data/Results"): os.makedirs("Data/Results")
    return a
# %%
def main(options=None):
    glVar.time_data_collect = time.time()
    if options is None:
        options = argument_parser().parse_args()       
    print("Testing")
    #Sets the folder locations to be tested 
    glVar.dateCode = str(datetime.now()).replace('.', '').replace(' ', '_').replace(':', '')
    if options.folder_test == "": glVar.folder_test = options.folder_train
    else: glVar.folder_test = options.folder_test
    if options.folder_train[-1] == "/": glVar.folder_train = options.folder_train[0:-1]
    else: glVar.folder_train = options.folder_train
    glVar.NNets = options.NNets
    glVar.col_param= options.col_param
    glVar.col_mods= options.col_mods
    glVar.num_points_train = options.num_points_train
    glVar.dtype = options.data_type
    glVar.NN_train = options.NN_train
    
    #var = os.getcwd().replace("//","/")
    options.NN_Hist_folder = os.path.join(os.getcwd(), "NN_Hist",  options.NN_Hist_folder)
    #options.NN_Hist_folder = options.NN_Hist_folder.replace("//","/")
    glVar.NN_Hist_folder = options.NN_Hist_folder.replace("//","/").replace("./", "")
    
    if glVar.folder_test =="" or glVar.folder_test == glVar.folder_train: 
        folders_test = getFolderList(glVar.folder_train)
        glVar.sep_train_test = False
    else: 
        folders_test = getFolderList(glVar.folder_test)
        print("Original Test Folders: ", folders_test)
        #Removes training folder from test set
        #folders_test = [x for x in folders_test if (glVar.folder_train[0:-1] not in x and "train" not in x)]
        folders_test = [x for x in folders_test if "train" not in x]
        if glVar.folder_train in folders_test: folders_test.remove(glVar.folder_train)

    #Creates list of logfiles.  
    #If the entry is a directory, all the files in the directory are appended to teh list
    li = []
    if len(options.logfile[0]) > 0: 
        for i in options.logfile:
            if os.path.isdir(i): glVar.logfile.extend( [s for s in glob.glob(i)])               
            else: glVar.logfile.append(i)
            #glVar.logfile.append(i)
        glVar.logfile = list(set(glVar.logfile)) 
        for j in glVar.logfile: li.append(pd.read_csv(j))
        #Concatentate pd dataframe. Removes entries with duplicate filenames
        glVar.testData = pd.concat(li, axis=0, ignore_index=True).drop_duplicates(subset=["filename"])
        glVar.testData["s1_mod"] = glVar.testData["s1_mod"].str.lower()
    else: sys.exit("Logfile not available. Please include an appropriate logfile location")

    #Gets list of files to exclude from the training and test data
    #options.range_train = list(np.asarray(options.range_train).astype(float))
    #options.range_test = list(np.asarray(options.range_test).astype(float))
    # glVar.exc_list_train = getExclusionList(range_param = options.range_param, range_arr = options.range_train, 
    #                             exc_param = options.exc_param, exc_arr = options.exc_train)
    # glVar.exc_list_test = getExclusionList(range_param = options.range_param, range_arr = options.range_test, 
    #                             exc_param = options.exc_param, exc_arr = options.exc_test)
    
    glVar.exc_list_test = []
    glVar.exc_list_train = []
    
    #Fuction to return a boolean value
    #Code from:
    #https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    def str2bool(v):
        if isinstance(v, bool):return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
        else: return False
    glVar.time_data_collect = np.round(time.time() - glVar.time_data_collect, 2)
    
    #This for loop run the main function for all folders in the "folders" array
    glVar.iter_f = 0
    for f in folders_test:
        glVar.iter_f = glVar.iter_f + 1
        glVar.time_start_OVH = time.time()
        glVar.folder_test = f 
        if not glVar.sep_train_test: glVar.folder_train = f
        if glVar.sep_train_test and glVar.iter_f > 1: glVar.NN_train = 0
        #print("\nTest Data: ", glVar.folder_test) 
        """"""                
        runTest(glVar.dateCode, datapoints = options.num_points, 
        samples = options.samples, num_iter = options.iter, 
        testAct = str2bool(options.test_act), options = options)
    print("Done")

#%% Runs main port of program
if __name__ == '__main__':
    main()
   
    
