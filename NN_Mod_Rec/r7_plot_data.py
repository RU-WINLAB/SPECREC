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
from datetime import datetime

#%% Setups up global variables
class glVar:
    temp = None
    dir_data = ""
    bin_info = None
    date_code = ""
    folder_data = ""
    
#%% Parses user input
def argument_parser():
    parser = ArgumentParser()
    parser.add_argument( 
        "--files-test", dest="files_test", type=str, nargs = '+',
        default= [""],
        help="Sets the training data folder location [default=%(default)r]")    
    parser.add_argument(
        "--samples", dest="samples", type=int, 
        default= 1000,
        help="Number of samples to collect [default=%(default)r]")
    parser.add_argument(
        "--offset", dest="offset", type=int, 
        default= 0,
        help="Offset [default=%(default)r]")
    parser.add_argument(
        "--plot-types", dest="plot_types", type=str, nargs='+', 
        default= ["MAG", "PSD", "IQ"],
        help='''Enter the list of plots to be created. \n
        MAG --> Magnitude vs Time \n
        PSD --> Power Spectral Density \n
        IQ --> IQ \n'''
        )
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
    parser.add_argument(
        "--plot-show", dest="plot_show", type=int, 
        default= 1,
        help='''Indicate whether or not you want to show the plot 
        0 --> Yes, 1 --> No
        [default=%(default)r]''')
    parser.add_argument(
        "--plot-save", dest="plot_save", type=int, 
        default= 1,
        help='''Indicate whether or not you want to save the plot 
        0 --> Yes, 1 --> No
        [default=%(default)r]''')
    return parser
#%%
def read_binary_iq(filename, count=1000, d_type = "int16", offset = 0, **kwargs):         
    if d_type == "int8": data_type = np.int8;
    elif d_type == "int16": data_type = np.int16;
    elif d_type == "int32": data_type = np.int32;
    elif d_type == "int64": data_type = np.int64;
    elif d_type == "uint8": data_type = np.uint8;
    elif d_type == "uint16": data_type = np.uint16;
    elif d_type == "uint32": data_type = np.uint32;
    elif d_type == "uint64": data_type = np.uint64;
    elif d_type == "float32": data_type = np.float32;
    elif d_type == "float64": data_type = np.float64; 
    else: data_type = np.float32
    if (filename.find(".txt") <= 0 and filename.find(".csv") <=0):
        with open(filename, 'rb') as f:
            f.seek(offset)    
            data = np.fromfile(filename, dtype=data_type, count = count + offset*2, **kwargs)
        data = data[offset*2:]
        f.close()
        dataiq = ((data[0::2] + data[1::2]*1j).astype(np.complex))
    else: 
        data = []; dataiq = [];
        print("Ignoring .csv or .txt file")
    #data = np.array(data/np.max(np.abs(data)), dtype = int)

    #print(data)
    return data, dataiq
#%%
def get_filelist_from_directory(loc_data):
    data = []
    for root, dirs, files in os.walk(loc_data):
        for f in files:
            data.append((root + "/" + f).replace("\\", "/").replace("//", "/"))
    return data
#%% Creates plot for IQ values based on user input
def plot_data(data, plot_types = ["IQ"], plot_name = "", plot_save = 1, plot_show = 1):
    i = 0
    for plot_type in plot_types:
        i = i+1
        plt.clf()
        plt.figure(i)
        if plot_type.upper() == "IQ": 
            I = data[0::2]; 
            Q = data[1::2];
            div = int(len(I)/4)
            glVar.temp = div
            plt.plot(I[0:div], Q[0:div], '*b');
            plt.plot(I[div+1:2*div], Q[div+1:2*div], '*b');
            plt.plot(I[2*div+1:3*div], Q[2*div+1:3*div], '*b');
            plt.plot(I[3*div+1:4*div], Q[3*div+1:4*div], '*b');
            plt.xlabel("I"); 
            plt.ylabel("Q")
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            plt.axhline(y = 0, linewidth = .75, color = 'k')
            plt.axvline(x = 0, linewidth = .75, color = 'k')
        elif plot_type.upper() == "MAG": 
            plt.plot(abs(data)); 
            plt.xlabel("Time"); 
            plt.ylabel("Magnitude")
        elif plot_type.upper() == "PSD": 
            plt.psd((data[0::2] + data[1::2]*1j).astype(np.complex),
                                         NFFT=1024, Fs=10000)    
        else: plt.plot(data)
    
        plt.title(plot_type)
        if plot_save == 1: plt.savefig(glVar.folder_data + '/' + plot_name+"_"+plot_type); print("Saving " + plot_name)
        if plot_show == 1: plt.show()
    return 0

#%%Main program
def main(options = None):
    if options is None:
        options = argument_parser().parse_args()        
    
    def sig_handler(sig=None, frame=None):
        sys.exit(0)
    #samples = 1000
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    glVar.date_code = str(datetime.now()).replace('.', '').replace(' ', '').replace(':', '').replace('-','')
    glVar.folder_data = "Data/Plots/" + glVar.date_code
    if not os.path.exists(glVar.folder_data): os.makedirs(glVar.folder_data)       
    
    for i in options.files_test: 
        if os.path.isdir(i): file_list = get_filelist_from_directory(i)
        else: file_list = [i]
        
        for file_test in file_list: 
            if (file_test.find(".txt") <= 0 and file_test.find(".csv") <=0):
                file_name = os.path.basename(file_test).split('.')
                data, data_complex = read_binary_iq(filename = file_test, count =options.samples,
                    d_type = options.data_type.lower(), offset = options.offset)
                plot_data(data = data, plot_types = options.plot_types, plot_name = file_name[0], 
                          plot_show = options.plot_show, plot_save = options.plot_save)
            else:  print("Ignoring .csv or .txt file")
    return 0

#%%
if __name__ == '__main__':
    main()

