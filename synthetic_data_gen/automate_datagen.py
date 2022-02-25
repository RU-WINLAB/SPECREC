#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 23:07:49 2019
Device: tina-mac2
Author: Tina Burns
School: Rutgers University - New Brunswick
Department: WINLAB
Advisor: Richard Martin
Date: Spring 2020

This code is used to automate tests with USRP Tests with different parameters for
frequency, modulation type, and gain.

"""
#Imports appropriate libraries
import sys, os, pandas as pd, random, signal
from datetime import datetime
sys.path.append('../drago')
import r3_calculate_snr as calculate_snr
from argparse import ArgumentParser
#os.system("echo hi")

 #%%
#Setups global variables to be accessed by all functions in the code
class glVar():    
    info_full = ""
    filename = ""
    path_folder = ""
    path_base = ""
    script_file = ""
    noise_calc_file = ""
    log_file = ""
    param_test = ""
    param_val = 0
    noise_seed = ""
    num_samp = "1000"
    samp_rate = "10M"
#%%
def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--script-file", dest="script_file", type=str, 
        default= "snr_additive.py",
        help="Sets location for the script file [default=%(default)r]")
    parser.add_argument(
        "--param-test", dest="param_test", type=str, default= "snr",  
        help='''Enter the parameter that will be tested. \r
        [snr --> snr]
        [atten --> attenuation]
        [gain --> signal gain]
        [noise --> noise gain]
        [Options [default=%(default)r]''')
    parser.add_argument(
        "--param-min", dest="param_min", type=int, 
        default= 0,
        help="Minimum value for test paramter [default=%(default)r]")
    parser.add_argument(
        "--param-max", dest="param_max", type=int, 
        default= 100,
        help="Maximum value for test paramter [default=%(default)r]")
    parser.add_argument(
        "--param-inc", dest="param_inc", type=int, 
        default= 10,
        help="Increment value for test paramter [default=%(default)r]")
    parser.add_argument(
        "--num-iter", dest="num_iter", type=int, 
        default= 1,
        help="Number of iterations [default=%(default)r]")
    parser.add_argument(
        "--samp-rate", dest="samp_rate", type=str, 
        default= "10M",
        help="Sample rate [default=%(default)r]")
    parser.add_argument(
        "--num-samp", dest="num_samp", type=str, 
        default= "100000000",
        help="Number of samples to collect [default=%(default)r]")
    parser.add_argument(
        "--modulations", dest="modulations", nargs = '+', type=str, 
        default= ["bpsk", "qpsk", "8psk", "16qam"],
        help="Number of samples to collect [default=%(default)r]")

    return parser
#%% General functions
def genDatecode():
    return str(datetime.now()).replace('.','').replace(' ', '').replace(':', '').replace('-', '')

def genFilename(mod="", param="", param_val="", rate = "", append = ""):
    filename = (param + str(param_val) + "_" + mod + "_r" + str(rate)+"_"+append)
    #print(filename)
    return filename
def genInfoDualMod(mod1, mod2, snr, attn2, attn1 = 0, gain = 1):
    info_base = " --sig1-freq 0 --sig2-freq 0 --sig1-bw 10M --sig2-bw 10M --attn1 0"
    return (info_base + " --modulation1 " + mod1 +  " --modulation2 " + mod2 + " --snr " + str(snr) + 
            " --attn1 " + str(attn1) + " --attn2 " + str(attn2))

def genInfoSingleMod(mod1, snr, gain = 1):
    info_base = " --sig1-freq 0 --sig1-bw 10M"
    return (info_base + " --modulation " + mod1 + " --snr " + str(snr) + " --gain " + str(gain) + " ")

def genInfoAdditive(mod1, snr, gain_sig = 1, gain_noise = 0):
    info_base = " --sig1-freq 0 --sig1-bw 10M"
    return (info_base + " --modulation " + mod1 + " --snr " + str(snr) + 
            " --noise-gain " + str(gain_noise) + " --signal-gain " + str(gain_sig) + " ")

def genInfoAdditiveSNR(mod1, snr, gain_sig = 1, gain_noise = 0):
    return (" --modulation " + mod1 + " --snr " + str(snr) + " ")

def genInfoNoise(): 
    return (" --samp-rate 4M --num-samp 15000000 ")

#%%
def moveFiles(param, myFile):
    glVar.path_folder = glVar.path_base + '/' + param + '/'
   
    if not os.path.exists(glVar.path_folder):
        os.makedirs(glVar.path_folder)    
    if os.path.exists(myFile):
        os.rename(myFile, glVar.path_folder + myFile)
    #Moves the clean signal files to a separate folder.
    if not os.path.exists(glVar.path_base +'/cleansig'):
        os.makedirs(glVar.path_base +'/cleansig')    
    if os.path.exists("cleansig_" + myFile):
        os.rename("cleansig_" + myFile, glVar.path_base +'/cleansig/'+ "cleansig_" + myFile)
    if os.path.exists("cleansig1_" + myFile):
        os.rename("cleansig1_" + myFile, glVar.path_base + '/cleansig/'+ "cleansig1_" + myFile)
    if os.path.exists("cleansig2_" + myFile):
        os.rename("cleansig2_" + myFile, glVar.path_base + '/cleansig/'+ "cleansig2_" + myFile)
#%%     
def runMods(modulation1, modulation2, num_sigs, snr, atten, num_iter = 1, gain_sig = 1,
            gain_noise = 1):
    #f = open(glVar.path_base + "/command_info.txt", "w+")
    for mod1 in modulation1:
        for mod2 in modulation2:
            if mod1 != mod2:    
                for i in range(1, int(num_iter) + 2):
                    print("\nStarting test...")            
                    glVar.filename = genFilename(mod = mod1, param = glVar.param_test, param_val =snr, rate = glVar.samp_rate, append = str(i))
                    glVar.noise_seed = str(random.randint(1, 2000)) 
                    #setsup up test information with appropriate parameters based on the script file
                    if glVar.script_file.find("snr") > -1: glVar.info_full =("python3 " + glVar.script_file + genInfoAdditiveSNR(mod1, snr, gain_sig = gain_sig, gain_noise = gain_noise))
                    elif glVar.script_file.find("add") > -1: glVar.info_full =("python3 " + glVar.script_file + genInfoAdditive(mod1, snr, gain_sig = gain_sig, gain_noise = gain_noise))                       
                    elif glVar.script_file.find("noise") > -1: glVar.info_full = "python3 " + glVar.script_file + genInfoNoise()
                    elif glVar.script_file.find("1tx") > -1: glVar.info_full =("python3 " + glVar.script_file + genInfoSingleMod(mod1, snr, gain_sig))
                    else: glVar.info_full = ("python3 " + glVar.script_file + genInfoDualMod(mod1, mod2, snr, atten))
                    
                    glVar.info_full = (glVar.info_full + " --filename " + glVar.filename  
                         + " --noise-seed " + glVar.noise_seed + " --samp-rate " + glVar.samp_rate
                         + " --num-samp " + glVar.num_samp)

                    #This section was placed here in order to attain different distortion in a  on appended file
                    #using the Channel and Noise dta generatin metions. 
                    if (glVar.script_file.find("noise") > -1 or glVar.script_file.find("add") > -1): n = 2
                    else: n = 2
                    
                    for j in range(1, n):
                        print(glVar.info_full)
                        #print("Sample " + str(j) + " of " + str(n))
                        os.system(glVar.info_full)
                    
                    if i == 1: param = "train"
                    else: param = glVar.param_test + str(glVar.param_val)
                    sinr1, sinr2 = calculate_snr.main(glVar.filename)
                    moveFiles(param, glVar.filename)
                    
                    print ("Test Complete for: iteration " + str(i))
                    #Writes file feature information to file
                    pd.DataFrame({  "filename": [glVar.filename],
                                     "num_sigs": [num_sigs],
                                     "iteration" : [i], 
                                     "s1_mod": [mod1], 
                                     "s2_mod": [mod2],
                                     "s1_freq": ["0"], 
                                     "s2_freq": ["0"],
                                     "s1_atten": [0], 
                                     "s2_atten": [atten],   
                                     "s1_sinr": [str(sinr1)], 
                                     "s2_sinr": [str(sinr2)],
                                     "snr": [snr],    
                                     #"samples": ["30000000"],
                                     "gain_sig": [str(gain_sig)],
                                     "gain_noise": [str(gain_noise)],
                                     "noise_seed": [glVar.noise_seed]
                                     }).to_csv(glVar.path_base + '/' + glVar.log_file + "_Logfile.csv", 
                                               mode = 'a', header = glVar.write_header)
                    glVar.write_header = False
                    #f.write(glVar.info_full + "\r\n")
    
    #f.close()

#%%
#Main code that is executed
def main(options=None):
    if options is None:
        options = argument_parser().parse_args()
            
    def sig_handler(sig=None, frame=None):
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    
    glVar.path_base = "Data/data_" + genDatecode()
    glVar.log_file = genDatecode()
    glVar.script_file = options.script_file
    glVar.param_test = options.param_test.lower()
    glVar.num_samp = options.num_samp
    glVar.samp_rate = options.samp_rate
    glVar.write_header = True   
    glVar.param_test = options.param_test
    num_sigs = 2 if glVar.script_file.find("2tx") > -1 else 1 

    modulation2 = options.modulations if num_sigs == 2 else ["n/a"]
    gain = 1
    snr = 100
    atten = 150
    
    if glVar.param_test == "atten":
        for atten in range (options.param_min, options.param_max +options.param_inc, options.param_inc):
            glVar.param_val = atten
            runMods(options.modulations, modulation2, num_sigs, snr, atten, options.num_iter,)
    elif glVar.param_test == "snr":
        for snr in range (options.param_min, options.param_max +options.param_inc, options.param_inc):
            glVar.param_val = snr
            runMods(options.modulations, modulation2, num_sigs, snr, atten, options.num_iter)
    elif glVar.param_test == "gain":
        for gain in range (options.param_min, options.param_max +options.param_inc, options.param_inc):
            glVar.param_val = gain
            runMods(options.modulations, modulation2, num_sigs, snr, atten, options.num_iter, 
                    gain_sig = gain/100)
    elif glVar.param_test == "noise":
        for noise in range (options.param_min, options.param_max + options.param_inc, options.param_inc):
            glVar.param_val = noise
            runMods(options.modulations , modulation2, num_sigs, snr, atten, options.num_iter, 
                    gain_noise = noise/100)
    else: print("Please enter a valid parameter for test.  Enter atten, snr," + 
                "or noise to indicate the parameter that will be adjusted")
            
#%%                
if __name__ == '__main__':
    main()
   
