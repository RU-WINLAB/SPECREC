# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:45:56 2020
@author: TINA
Program to make several plots based on data and user requests
"""
#%% imports the appropriate libraries
#!/usr/bin/env python3
import numpy as np, signal, sys, pandas as pd, os
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scipy import stats

#%% Setups up global variables
class glVar:
    temp = None
    dir_data = ""
    bin_info = None
    
#%% Parses user input
def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--file-test", dest="file_test", type=str, 
        default= "C:/Users/TINA/OneDrive - Rutgers University/Rutgers/Research/SDR/Data/2020-09-05_wifi/rx_1m_2412e6Hz_1MCS_node18-2+node18-1_float32.dat",
        help="Sets the training data folder location [default=%(default)r]")    
    parser.add_argument(
        "--samples", dest="samples", type=int, 
        default= 1000,
        help="Number of samples to collect [default=%(default)r]")
    parser.add_argument(
        "--num-points", dest="num_points", type=int, 
        default= 1,
        help="Number of datapoints [default=%(default)r]")
    parser.add_argument(
        "--data-type", dest="data_type", type=str, 
        default= "float32",
        help='''Sets the data type  [default=%(default)r]
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
def read_binary_iq(fname, samples=1, num_points = 1, pos_sample = 0, 
                   d_type = "int16", **kwargs):         
    if d_type == "int8": data_type = np.int8;
    elif d_type == "int16": data_type = np.int16; num_bits = 16;  
    elif d_type == "int32": data_type = np.int32; num_bits = 32;
    elif d_type == "int64": data_type = np.int64; num_bits = 64;
    elif d_type == "uint8": data_type = np.uint8; num_bits = 8;
    elif d_type == "uint16": data_type = np.uint16; num_bits = 16;
    elif d_type == "uint32": data_type = np.uint32; num_bits = 32;
    elif d_type == "uint64": data_type = np.uint64; num_bits = 64;
    elif d_type == "float32": data_type = np.float32; num_bits = 32;
    elif d_type == "float64": data_type = np.float64; num_bits = 64;
    else: data_type = np.float32; num_bits = 32;
    
    #Exits program if there are not enough samples in the file
    samples_avail = (os.path.getsize(fname)*8)/(num_bits*num_points) 
    #- pos_sample/samples)   
    #print("Available Samples: ", samples_avail)
    if samples_avail*num_points - pos_sample <= samples*num_points:
        print ("Not enough datapoints in file.") 
        print("Please decrease the sample size or decrease the datapoint starting position")
        print ("Samples requested: ", samples)
        print ("Samples available: ", samples_avail)
        print("Datapoint starting position: ", pos_sample)
        sys.exit("")
    if fname != ".DS_Store": #Ignores .DS_Store file
        data = np.fromfile(fname, dtype=d_type, count = samples*num_points + pos_sample,  **kwargs)
        #data = data[100:]
        glVar.temp = data
        #print("Data point start position", pos_sample)
    return data[pos_sample:]


#%%Main program
def main(options = None):
    if options is None:
        options = argument_parser().parse_args()        
    
    def sig_handler(sig=None, frame=None):
        sys.exit(0)
    #samples = 1000
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    
    file_name = os.path.basename(options.file_test).split('.')
    data = read_binary_iq(filename = options.file_test, 
        count =options.samples, d_type = options.data_type.lower())

    return 0

#%%
if __name__ == '__main__':
    main()
   