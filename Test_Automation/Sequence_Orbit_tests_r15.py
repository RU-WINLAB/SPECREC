#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 23:07:49 2019
Device: tina-mac2

Author: Tina Burns
School: Rutgers University - New Brunswick
Department: WINLAB
Advisor: Richard Martin
Date: Fall 2021

This code is used to automate tests with USRP Tests with different parameters for
frequency, modulation type, and gain.

"""
#%% Imports appropriate libraries
import sys, threading, time, os, signal, csv, ntpath
from argparse import ArgumentParser
from datetime import datetime
#from Queue import Queue
#os.system("echo hi")
#%% Setups global variables to be accessed by all functions in the code
class glVar():    
    #This was the full line used for test purposes
    #tx_run =  "./gnuradio/gr-digital/examples/narrowband/benchmark_tx.py -r 100000 --tx-amplitude .2 -M 0.65 -f 500M -m cpm --tx-gain 5"
    #rx_run_drift = " ./uhd/host/build/examples/rx_ascii_art_dft --freq 2e9 --gain 10 --rate 5e6 --frame-rate 10 --ref-lvl -30 --dyn-rng 70"
    #rx_run_samp_file = './uhd/host/build/examples/rx_samples_to_file --args="addr=192.168.40.2" --rate 5e6 --duration 100 --gain 10 --type short --freq 2e9 --file bpsk.dat'
    temp = None 
    test_info_file = None
    test_info = []
    path_base = "Data"
    path_data = ""
    datecode = ""
    tx_datafile_name = ""
    rx_datafile_name = ""
    tx_datafile_list = []
    rx_node = ""
    tx_node = ""
    filenaming = "0"
    filenaming_id = ""
    files_copy_init = 1 
    options = None
    rx_wait_time = 0
#%% Parses user input
def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--tx-script", dest="tx_script", type=str, 
        default= "./uhd/host/build/examples/tx_samples_from_file ",
        help="Sets location for the transmitter script file [default=%(default)r]")
    parser.add_argument(
        "--rx-script", dest="rx_script", type=str, 
        default= "./uhd/host/build/examples/rx_samples_to_file ",
        help="Sets location for the receiver script file [default=%(default)r]")
    parser.add_argument(
        "--tx-gain", dest="tx_gain", type=str, nargs = '+', 
        default= ["15"],
        help="Sets the gain for the transmitter [default=%(default)r]")
    parser.add_argument(
        "--rx-gain", dest="rx_gain", type=str, nargs = '+', 
        default= ["15"],
        help="Sets the gain for the receiver [default=%(default)r]")
    parser.add_argument(
        "--tx-node", dest="tx_node", type=str, nargs = '+', 
        default= ["node1-1"],
        help="Sets the transmitter node [default=%(default)r]")
    parser.add_argument(
        "--rx-node", dest="rx_node", type=str, nargs = '+', 
        default= ["node1-2"],
        help="Sets the receiver node [default=%(default)r]")
    parser.add_argument(
        "--tx-dev", dest="tx_dev", type=str, nargs = '+', 
        default= ["x310"],
        help="Enter the type of device for the transmitters [default=%(default)r]")
    parser.add_argument(
        "--rx-dev", dest="rx_dev", type=str, nargs = '+', 
        default= ["x200"],
        help="Enter the type of device for the transmitters y[default=%(default)r]")
    parser.add_argument(
        "--tx-addr", dest="tx_addr", type=str, nargs = '+', 
        default= ["0"],
        help="Enter the ip addresses of transmitters [default=%(default)r]")
    parser.add_argument(
        "--rx-addr", dest="rx_addr", type=str, nargs = '+', 
        default= ["0"],
        help="Enter the ip addresses of the receivers y[default=%(default)r]")
    parser.add_argument(
        "--bitrate", dest="bitrate", type=str,  nargs = '+',
        default= ["20e6"],
        help="Birate of UHD Device [default=%(default)r]")
    parser.add_argument(
        "--dtype", dest="dtype", type=str, 
        default= "float",
        help="Sets numeric data type of the input and output files [default=%(default)r]")
    parser.add_argument(
        "--modulations", dest="modulations", type=str, nargs = '+', 
        default= ["bpsk", "qpsk", "8psk", "16qam"],
        help="Types of modulations [default=%(default)r]")
    parser.add_argument(
        "--freq", dest="freq", type=str, nargs = '+', 
        default= ["2412e6"],
        help="Frequency [default=%(default)r]")
    parser.add_argument(
        "--rx-duration", dest="rx_duration", type=str, 
        default= "1",
        help="Length of time to run receiver [default=%(default)r]")
    parser.add_argument(
        "--filenaming", dest="filenaming", type=str, 
        default= "0",
        help='''
        Specifies the tNames files same as the transmitter input data file. Retains folder structureype of naming convention to use for the output files: \n\r 
        0 --> Standard naming convention based on test conditions (Receiver input). \n\r 
        1 --> Names files same as the transmitter input data file. Ignores folder structure.\n\r
        2 --> Names files same as the transmitter input data file. Retains folder structure \n\r 
        [default=%(default)r]"
        ''')
    parser.add_argument(
        "--filenaming-ids", dest="filenaming_ids", type=str, nargs = '+',
        default= [""],
        help='''Specific the unique identifiers to append to the end of a file. 
        To run the receiver loop multiple times, give each loop a unique id. 
        [default=%(default)r]"
        ''')
    parser.add_argument(
        "--files-test", dest="files_test", type=str,  nargs = '+',
           default= [""],
        help="files to be tested [default=%(default)r]")
    parser.add_argument(
        "--files-test-struct", dest="files_test_struct", type=str, 
        default= "0",
        help='''
        Specifies the structure of the input files
        0 --> List of individual files 
        1 --> List of folders containing files
        2 --> csv file with list of files 
        [default=%(default)r]"
        ''')
    parser.add_argument(
        "--files-copy-dest", dest="files_copy_dest", type=str, 
        default= "~",
        help='''
        Specifies where to move the files to
        [default=%(default)r]"
        ''')
    parser.add_argument(
        "--files-copy", dest="files_copy", type=int, 
        default= 0,
        help='''
        Specifies whether or not to move the files
        0 --> Do not copy files
        1 --> Copy files to a place on the 
        2 --> Copy files to a gdrive folder
        [default=%(default)r]"
        ''')
    parser.add_argument(
        "--tx-agc", dest="tx_agc", type=int, 
        default= 0,
        help='''
        Specifies whether or not to enable the transmitter AGC. 
        As a note the AGC funciton on only works with B210s
        0 --> No Tx AGC
        1 --> Enable Tx AGC 
        [default=%(default)r]"
        ''')
    parser.add_argument(
        "--rx-agc", dest="rx_agc", type=int, 
        default= 0,
        help='''
        Specifies whether or not to enable the transmitter AGC. 
        As a note the AGC funciton on only works with B210s
        0 --> No Rx AGC
        1 --> Enable Rx AGC 
        [default=%(default)r]"
        ''')
    return parser
#%%
def gen_file_list_from_directory(loc_data):     
    #a = [*set([loc_data + "/" + p for p in os.listdir(loc_data)])]
    data = []
    for root, dirs, files in os.walk(loc_data):
        for f in files:
            if (f.lower().find("log") > -1 or f.lower().find(".txt") > -1 or 
                f.lower().find(".csv") > -1) : print(f)
            else: data.append((root + "/" + f).replace("\\", "/"))
    return data
#%%
def gen_file_list_from_file(list_file):
    if len(list_file) == 1: list_file = list_file[0] #Converts list to item
    with open(list_file, newline='') as f:
        reader = csv.reader(f)
        data = list(reader) 
        #print("File info: ", data)
        return data
#%%     
def make_data_folders(tx_nodes, rx_nodes, base = "", filenaming = "0", file_dir_list = []):   
    for rx in rx_nodes:
        for tx in tx_nodes:
            my_path = (base + "/tx-" + tx + "_rx-" + rx)
            os.system("ssh -o 'StrictHostKeyChecking no' root@" + rx +" mkdir -p -m755 " + my_path)
        
            li = []
            if filenaming == "2":
                for i in file_dir_list:
                    if len(i) == 1: i = i[0] #Converts list to item
                    li.append(ntpath.basename(os.path.dirname(i))) 
                for j in list(set(li)):
                    os.system("ssh -o 'StrictHostKeyChecking no' root@" + rx + " mkdir "+ my_path + "/" + j)
    
    if glVar.options.files_copy == 1: 
       print("Creating scp directory")
       os.system("mkdir " + glVar.options.files_copy_dest)    
    elif glVar.options.files_copy == 2: 
        print("Creating rclone directory")
        os.system("ssh -o 'StrictHostKeyChecking no' root@" + rx_nodes[0] + " rclone mkdir -v "  + glVar.options.files_copy_dest)

    
    return 0

#%% Generates filename for receiver output file
def gen_fileName(typ = "rx", freq="", rate="", rx_node = "", tx_node ="", tx_gain="", rx_gain = "", dtype="", modulation = "", fname = ""):    
    
    if fname == "": fname_out = ("./" + glVar.path_data + "/"+ typ + "_mydata.dat")
    else:
        if glVar.filenaming == "1": #For naming files based on tx files and ignoring folder structure
            fname_out = ("./" + glVar.path_data + "/"+ ntpath.basename(fname))
        elif glVar.filenaming == "2": #For naming files based on tx files and while keeping their folder structure
            fname_out = ("./" + glVar.path_data + "/" + ntpath.basename(os.path.dirname(fname))
            + "/" + ntpath.basename(fname))
        else: #for naming files based on testing conditions
            if fname.upper().find("MCS" )>=0: mod = "MCS-" + fname.split("MCS")[0][-1]
            elif fname.upper().find("BPSK" )>=0: mod = "BPSK"
            elif fname.upper().find("QPSK")>=0: mod = "QPSK"
            elif fname.upper().find("8PSK" )>=0: mod = "8PSK"
            elif fname.upper().find("16QAM" )>=0: mod = "16QAM"
            elif fname.upper().find("64QAM" )>=0: mod = "64QAM"
            elif fname.upper().find("GMSK" )>=0: mod = "GMSK"
            elif fname.upper().find("GFSK" )>=0: mod = "GFSK"
            else: mod = modulation

            fname_out = ("./" + glVar.path_data + "/"+ "tx-" + tx_node + "_rx-" + 
            rx_node + "_gTx" + tx_gain  + "_gRx" + rx_gain + "_rate" + rate
            + "_m" + mod + "_f" + freq+"Hz_"+ dtype +"_" + glVar.filenaming_id+ ".dat")
    return fname_out

#%% Check lengths and sets to the appropriate length if necesssary. 
def check_list_length(arr_sample, arr_test):
    if len(arr_sample) == len(arr_test): arr_out = arr_test
    elif len(arr_sample) > 1 and len(arr_test) == 1: arr_out = arr_test*len(arr_sample)
    else: sys.exit("The arrays are not of the proper lengths. ")
    return arr_out

#%% Gets arguments for USRP Device
def get_args(addr, dev):
    #Get device type
    if dev.upper().find("N2") > -1: a = "type=usrp2"
    elif dev.upper().find("B2") > -1: a = "type=b200"
    elif dev.upper().find("E1") > -1: a = "type=e100"
    elif dev.upper().find("E3") > -1: a = "type=e3x0"
    elif dev.upper().find("X3") > -1: a = "type=x300"
    else: a = "type=usrp2"
    #Gets address information
    if addr.upper() == "RIO0": a= a+",resource=RIO0"
    elif addr == "0": a = a
    else: a = a + ",addr=" + addr
    args = '--args=' + '"' + a + '"'        
    print(args)
    return args

#%% This is used to run the transmiter 
def run_tx(node, script, addr, dev, tx_file, dtype, freq, rate, gain, agc = 0):
    print("Starting tx") 
    time.sleep(1)
    tx_info = ("ssh -o 'StrictHostKeyChecking no' root@" + node + " " + script + get_args(addr, dev) +
                " --file " + tx_file + " --type " +  dtype + " --rate " + rate +
                " --freq " + freq + " --gain "+ gain + " --repeat")    
    if agc == 1: tx_info = tx_info + " --agc"
    print(tx_info)
    glVar.tx_datafile_name = tx_file
    glVar.test_info_file.write("\n\rTx info: \n\r" + tx_info)
    os.system(tx_info)
    time.sleep(1)      

#%% This is used to run the transmiter      
def run_rx(tx_node, rx_node, script, addr, dev, rx_file_out, dtype, freq, rate, tx_gain, rx_gain, 
          duration = 1, agc = 0, rx_wait_time = 1):
    print("\n\rStarting the rx at " + rx_node)
    time.sleep(float(rx_wait_time) + float(duration))
    print(rx_file_out)
    #Setsup the file naming details
    glVar.path_data = (glVar.path_base + "/tx-" + tx_node + "_rx-" + rx_node)
    fname = gen_file_name(freq = freq, typ = "rx", rate = rate, tx_gain = tx_gain, rx_gain = rx_gain,
                        rx_node = rx_node, tx_node=tx_node, dtype = dtype, fname = rx_file_out)
    #Corrects bugs file extension names
    fname = fname.replace('~', '.').replace("//", "/")
    #Output information the screen and a text file
    sys_info_rx = ("ssh -o 'StrictHostKeyChecking no' root@" + rx_node + " " + script + get_args(addr, dev) + 
        " --file " + fname + " --type " +  dtype + " --rate " + rate +
        " --freq " + freq + " --gain "+ rx_gain + " --duration "  + duration)    
    if agc == 1: sys_info_rx = sys_info_rx + " --agc"
    print(sys_info_rx)
    glVar.test_info_file.write("\n\rRx info: \n\r" + sys_info_rx)   
    #Run rx
    os.system(sys_info_rx)
    copy_files( dest_in = fname, rx = rx_node)   
    return 0
#%% Stops tx and rx script if they are still running     
def close_uhd_handle(node, script_id):
    os.system("ssh -o 'StrictHostKeyChecking no' root@" + node + " /usr/bin/pkill --signal SIGINT " + script_id)
    return 0

#%% Copies files to appropriate destination
def copy_files(dest_in = "", dest_path = "", rx = ""):    
    dest_out = os.path.join(glVar.options.files_copy_dest, os.path.basename(os.path.dirname(dest_in)))    
    if glVar.options.files_copy != 0: 
        dest_out.replace("./", "")
        #os.system('scp Data/logs/d' + glVar.datecode + '_command_info.txt root@' + rx + ':' +glVar.path_base)
        print("Copying files")
        # Uses scp to copy files to specifed location
        if glVar.options.files_copy == 1: 
            sys_info_cp = ("scp -r root@" + rx + ":" + dest_in  + " " + dest_out)
        # Uses rclone to copy files to specifed location on Google Drive          
        if glVar.options.files_copy  == 2: 
            sys_info_cp = ("ssh -o 'StrictHostKeyChecking no' root@" + rx + " rclone copy -v "  
                + dest_in + " " + dest_out)
        print(sys_info_cp)
        os.system(sys_info_cp)
        print("Removing file " + dest_in + " from " + rx)
        os.system("ssh -o 'StrictHostKeyChecking no' root@" + rx + " rm -fr " + dest_in)
    return 0
#%% This function runs the transmitter as threads so that both can be executed 
    #simultaneously
def run_link_threads(tx_nodes, rx_nodes, tx_script, tx_addrs, tx_dev, rx_script, rx_addrs, rx_dev, tx_gains, rx_gains,
                   tx_file,rx_file_out, dtype, freq, rate, rx_duration, tx_agc = 0, rx_agc = 0, filenaming_ids = [""]):  
    #files array with blanks
    rx_node = ["", "", "", "", "", "", "", ""];
    rx_node[0:len(rx_nodes)] = rx_nodes
    rx_addr = ["", "", "", "", "", "", "", ""];
    rx_addr[0:len(rx_addrs)] = rx_addrs
    
    for tx_node, tx_addr in zip(tx_nodes, tx_addrs):
        glVar.tx_node = tx_node
        for tx_gain in tx_gains:
            first_run = True
            tx_thread = threading.Thread(target = run_tx, args = (tx_node, tx_script, tx_addr, tx_dev, tx_file, 
                dtype, freq, rate, tx_gain, tx_agc), daemon= True)
            tx_thread.start()
        
            for name in filenaming_ids:
                glVar.filenaming_id = name
                #for rx_node, rx_addr in zip (rx_nodes, rx_addrs):
                for rx_gain in rx_gains:
                    #Adds a delay to esnure that the tx has time to start up 
                    print("\n\rPausing before starting the receivers")
                    if first_run: time.sleep(10); glVar.rx_wait_time = 10; thread_timeout = 100
                    else: time.sleep(1); glVar.rx_wait_time = 1; thread_timeout = 60
                
                    rx_thread0 = threading.Thread(target = run_rx, args = (tx_node, rx_node[0], rx_script, rx_addr[0], rx_dev,
                        rx_file_out, dtype, freq, rate, tx_gain, rx_gain, rx_duration, rx_agc), daemon = True)
                    rx_thread1 = threading.Thread(target = run_rx, args = (tx_node, rx_node[1], rx_script, rx_addr[1], rx_dev,
                        rx_file_out, dtype, freq, rate, tx_gain, rx_gain, rx_duration, rx_agc), daemon = True)
                    rx_thread2 = threading.Thread(target = run_rx, args = (tx_node, rx_node[2], rx_script, rx_addr[2], rx_dev,
                        rx_file_out, dtype, freq, rate, tx_gain, rx_gain, rx_duration, rx_agc), daemon = True)
                    rx_thread3 = threading.Thread(target = run_rx, args = (tx_node, rx_node[3], rx_script, rx_addr[3], rx_dev,
                        rx_file_out, dtype, freq, rate, tx_gain, rx_gain, rx_duration, rx_agc), daemon = True)
                    rx_thread4 = threading.Thread(target = run_rx, args = (tx_node, rx_node[4], rx_script, rx_addr[4], rx_dev,
                        rx_file_out, dtype, freq, rate, tx_gain, rx_gain, rx_duration, rx_agc), daemon = True)
                    rx_thread5 = threading.Thread(target = run_rx, args = (tx_node, rx_node[5], rx_script, rx_addr[5], rx_dev,
                        rx_file_out, dtype, freq, rate, tx_gain, rx_gain, rx_duration, rx_agc), daemon = True)
                    rx_thread6 = threading.Thread(target = run_rx, args = (tx_node, rx_node[6], rx_script, rx_addr[6], rx_dev,
                        rx_file_out, dtype, freq, rate, tx_gain, rx_gain, rx_duration, rx_agc), daemon = True)
                    rx_thread7 = threading.Thread(target = run_rx, args = (tx_node, rx_node[7], rx_script, rx_addr[7], rx_dev,
                        rx_file_out, dtype, freq, rate, tx_gain, rx_gain, rx_duration, rx_agc), daemon = True)
                  
                    if rx_node[0] != "": rx_thread0.start()
                    if rx_node[1] != "": rx_thread1.start()
                    if rx_node[2] != "": rx_thread2.start()
                    if rx_node[3] != "": rx_thread3.start()
                    if rx_node[4] != "": rx_thread4.start()
                    if rx_node[5] != "": rx_thread5.start()
                    if rx_node[6] != "": rx_thread6.start()
                    if rx_node[7] != "": rx_thread7.start()
                    
                    #This waits for the rx_thread to end or for the program to timeout
                    for i in range(thread_timeout + int(float(rx_duration))):
                        if (rx_thread0.isAlive() or  rx_thread3.isAlive() or rx_thread6.isAlive() or
                            rx_thread1.isAlive() or rx_thread4.isAlive() or rx_thread7.isAlive() or 
                            rx_thread2.isAlive() or rx_thread5.isAlive()): 
                            time.sleep(1);
                        else: break    
                    
                    time.sleep(glVar.rx_wait_time + float(rx_duration)) 
                    if rx_node[0] != "": close_uhd_handle(rx_node[0], "rx_samples")                    
                    if rx_node[1] != "": close_uhd_handle(rx_node[1], "rx_samples")
                    if rx_node[2] != "": close_uhd_handle(rx_node[2], "rx_samples")
                    if rx_node[3] != "": close_uhd_handle(rx_node[3], "rx_samples")
                    if rx_node[4] != "": close_uhd_handle(rx_node[4], "rx_samples")
                    if rx_node[5] != "": close_uhd_handle(rx_node[5], "rx_samples")
                    if rx_node[6] != "": close_uhd_handle(rx_node[6], "rx_samples")
                    if rx_node[7] != "": close_uhd_handle(rx_node[7], "rx_samples")
            close_uhd_handle(tx_node, "tx_samples")             

    print("Thread complete")
                   
#%% Main program
def main(options = None):
    if options is None:
        options = argument_parser().parse_args()   
    
    def sig_handler(sig=None, frame=None):
        sys.exit(0)    
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
        
    if not os.path.exists("Data"): os.makedirs("Data")
    if not os.path.exists("Data/logs"): os.makedirs("Data/logs")
    glVar.datecode = str(datetime.now()).replace('.', '').replace(' ', '').replace(':', '').replace('-', '')
    glVar.test_info_file = open( "Data/logs/d" + glVar.datecode + "_command_info.txt", "w+")
    glVar.filenaming = options.filenaming
    glVar.options = options
    glVar.path_base = "Data/d" + glVar.datecode
    print(options.files_test)
    if options.files_test_struct == "0": glVar.tx_datafile_list = [options.files_test]
    elif options.files_test_struct == "1": glVar.tx_datafile_list = gen_file_list_from_directory(options.files_test)
    elif options.files_test_struct == "2": glVar.tx_datafile_list = gen_file_list_from_file(options.files_test)
    else: glVar.tx_datafile_list = [options.files_test]
    print("File List", glVar.tx_datafile_list)
    
    #Make folders
    make_data_folders(options.tx_node, options.rx_node, glVar.path_base, filenaming = glVar.filenaming, 
                                               file_dir_list= glVar.tx_datafile_list);
    
    #Sets arrays to proper length for zip function
    options.rx_addr = check_list_length(options.rx_node, options.rx_addr)
    options.tx_addr = check_list_length(options.tx_node, options.tx_addr)
    
    #initalizes files folders
    
    
    for tx_datafile in glVar.tx_datafile_list:
        if not (".txt" in tx_datafile and ".csv" in tx_datafile): #ignores txt and csv files
            for freq in options.freq:
                for rate in options.bitrate:
                    print("\nStarting tests...")
                    if len(tx_datafile) == 1: tx_datafile = tx_datafile[0] #Converts list to item
                    # if (.lower().find("log") > -1 or f.lower().find(".txt") > -1 or 
                    #     f.lower().find(".csv") > -1) : print(f)                    
                    run_link_threads(tx_nodes= options.tx_node, rx_nodes=options.rx_node, tx_script = options.tx_script, 
                    tx_addrs = options.tx_addr, tx_dev =options.tx_dev[0], rx_script = options.rx_script, 
                    rx_addrs = options.rx_addr, rx_dev = options.rx_dev[0], tx_gains=options.tx_gain,
                    rx_gains=options.rx_gain, tx_file = tx_datafile, rx_file_out = tx_datafile, dtype = options.dtype, 
                    freq=freq, rate = rate, rx_duration = options.rx_duration,
                    rx_agc=options.rx_agc, tx_agc=options.tx_agc, filenaming_ids = options.filenaming_ids) 
                    print ("Test Complete ")

    glVar.test_info_file.close()
    #glVar.datecode = str(datetime.now()).replace('.', '').replace(' ', '').replace(':', '').replace('-', '')
    print("exiting program")
    return 0

#%%
if __name__ == '__main__':
    main()
    
