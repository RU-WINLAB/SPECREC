# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:06:51 2021

@author: TINA
This code is used to sequence the orbit setup process
"""
#%%
import signal, sys, os, time, subprocess
import platform
from argparse import ArgumentParser

#%% Setups up global variables
class glVar:
    temp = None
    # Saved infor
    #image = "docmis2001-node-node1-2.sb2.orbit-lab.org-2021-02-25-16-23-30.ndz"
    # contains gnuradio and uhd does not work with b210s
    image = "usrpcal_2020-02-24.ndz" #For x310s on the grid
    #for usrp b210.  Does not have gnuradio
    #image = "baseline-uhd-3_13.ndz" #for B210s
#%% Parses user input
def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--node-image", dest="node_image", type=str, nargs = '+',
        default= [glVar.image],
        help="Sets location for the script file [default=%(default)r]")
    parser.add_argument(
        "--nodes", dest="nodes", type=str, nargs = '+', 
        default= ["node1-1", "node1-2"],
        help="Nodes that need to be calibrated [default=%(default)r]")
    parser.add_argument(
        "--dev-type", dest="dev_type", type=str, nargs = '+', 
        default= ["x310"],
        help="Type of USRP Device [default=%(default)r]")
    parser.add_argument(
        "--software", dest="software", type=str, nargs = '+', 
        default= ["octave", "singularity"],
        help="ip addresses of the devies that will be used [default=%(default)r]")
    parser.add_argument(
        "--dev-addr", dest="dev_addr", type=str, nargs = '+', 
        default= ["192.168.40.2"],
        help="ip addresses of the devies that will be used [default=%(default)r]")
    parser.add_argument(
        "--node-addr", dest="node_addr", type=str, nargs = '+', 
        default= ["192.168.40.1"],
        help="ip addresses of the nodes that will be used [default=%(default)r]")
    parser.add_argument(
        "--cmd", dest="cmd", type=str, nargs = '+', 
        default= [" ls ", " ls -ltrh "],
        help="Command to be run on the node[default=%(default)r]")
    parser.add_argument(
        "--cmd-type", dest="cmd_type", type=str, nargs = '+', 
        default= ["L"],
        help='''Type of command actions
        L--> Linux Command
        P2 --> Run a python2 script
        P3 --> Run a python3 script
        [default=%(default)r]''')
    parser.add_argument(
        "--files-test", dest="files_test", type=str, nargs = '+', 
        default= ["Code", "Data"],
        help="Files or folders to be copied onto the node [default=%(default)r]")
    parser.add_argument(
        "--files-copy-dest", dest="files_copy_dest", type=str, 
        default= "",
        help='''
        Specifies where to move the files to
        [default=%(default)r]"
        ''')
    parser.add_argument(
        "--files-copy", dest="files_copy", type=str, 
        default= "1a",
        help='''
        Specifies whether or not to move the files
        0 --> Do not copy files
        1a --> Copy files from the node to a directory
        1b --> Copy files from the directory to the node
        2a --> Copy files from a node to a gdrive folder-r CDsdfafdasdfdsf
        [default=%(default)r]"
        ''')
    parser.add_argument(
        "--actions", dest="actions", type=str, nargs = '+', 
        default= ["load-node",  "power-on-node", "load-dev", "power-cycle-node",
                  "config-eth"],
        help='''Action that needs to be performed
        load-node --> loads images on nodes
        config-eth --> ethernet configuration
        power-node --> power on nodes
        load-dev --> load image on USRP Device
        run-cmd --> Runs specified commands on node
        copy-files --> copies files to node using scp
        [default=%(default)r]''')
    return parser
#%% Configures the ethernet settings for the nodes
def conf_eth(nodes, node_addr):
    #if len(dev_ip) == 1: print(1)
    for i, j in zip(nodes, node_addr):
        ping(i)
        #Configures the ethernet for the USRPs
        print("Configuring ethernet settings for " + i)
        os.system("ssh -o 'StrictHostKeyChecking no' root@" + i + " ifconfig eth2 down")
        os.system("ssh -o 'StrictHostKeyChecking no' root@" + i + " ifconfig eth2 "+ j +" netmask 255.255.255.0 mtu 9000 up")
        #os.system("ssh -o 'StrictHostKeyChecking no' root@" + i + " sudo sysctl -w net.core.rmem_max=576000 ")
        os.system("ssh -o 'StrictHostKeyChecking no' root@" + i + " sudo sysctl -w net.core.rmem_max=33554432")
        #os.system("ssh -o 'StrictHostKeyChecking no' root@" + i + " sudo sysctl -w net.core.wmem_max=288000")
        os.system("ssh -o 'StrictHostKeyChecking no' root@" + i + " sudo sysctl -w net.core.wmem_max=576000") 

#%% Configures usrp
def get_args(nodes, dev_addr, dev_type):
    print("Constructing USRP Arguments")

    if "orbit":    
        B210s = ["node3-2", "node3-2","node3-2","node3-2","node3-2","node3-2",
                 "node3-2","node3-2","node3-2","node3-2","node3-2","node3-2","node3-2",
                 "node3-2","node3-2","node3-2","node3-2","node3-2","node3-2","node3-2",
                 "node3-2","node3-2","node3-2","node3-2","node3-2","node3-2","node3-2",
                 "node3-2","node3-2","node3-2","node3-2","node3-2","node3-2","node3-2",]
    
    for i, j, k in zip(nodes, dev_addr, dev_type):
        print("Loading UHD image on " + i + " at " + j)
        k= k.upper()
        if k.find("N2") > -1: dev = "usrp2"
        elif k.find("B2") > -1: dev = "b200"
        elif k.find("E1") > -1: dev = "e100"
        elif k.find("E3") > -1: dev = "e3x0"
        elif k.find("X3") > -1: dev = "x300"
        else: dev = "usrp2"
        os.system("ssh -o 'StrictHostKeyChecking no' root@" + i + ' "/usr/local/bin/uhd_image_loader" --args="type=' +
                  dev + ',addr=' + j + ',reset"')
        os.system("omf tell -a offh -t " + i)
        time.sleep(10)
#%% Setups array to be the same length
def setArrLength(arr1, arr2):
    arr1_out = []
    arr2_out = []
    if len(arr1) == 1 and len(arr2) > 1: arr1_out = [arr1[0]*len(arr2)]; arr2_out = arr2
    elif len(arr1) > 1 and len(arr2) == 1: arr2_out = [arr2[0]*len(arr1)]; arr1_out = arr1
    elif len(arr1)== 1 and len(arr2) == 1: arr1_out = arr1; arr2_out= arr2;
    else: sys.exit("Arrays are not the proper lengths")
    return arr1_out, arr2_out
#%% Configures usrp
def load_dev_image(nodes, dev_addr, dev_type):
    print("Configuring USRP Device")
    #if len(dev_addr) == 1: print(1)
    for i, j, k in zip(nodes, dev_addr, dev_type):
        print("Loading UHD image on " + i + " at " + j)
        k= k.upper()
        if k.find("N2") > -1: dev = "usrp2"
        elif k.find("B2") > -1: dev = "b200"
        elif k.find("E1") > -1: dev = "e100"
        elif k.find("E3") > -1: dev = "e3x0"
        elif k.find("X3") > -1: dev = "x300"
        else: dev = "usrp2"
        #Sets device arguments
        if j.upper() == "RIO0": args = '"type=' + dev + ',resource=RIO0"'
        elif j == "0": args = '"type=' + dev + '"'
        else: args = '"type=' + dev + ',addr=' + j + '"'
        
        os.system("ssh -o 'StrictHostKeyChecking no' root@" + i + " /usr/local/lib/uhd/utils/uhd_images_downloader.py")
        os.system("ssh -o 'StrictHostKeyChecking no' root@" + i + " uhd_find_devices")
        if j.upper() == "RIO0": os.system("ssh -o 'StrictHostKeyChecking no' root@" + i  + ' uhd_usrp_probe  --args="resource=RIO0"')
        #print("ssh -o 'StrictHostKeyChecking no' root@" + i + ' /usr/local/bin/uhd_image_loader --args=' + args)
        os.system("ssh -o 'StrictHostKeyChecking no' root@" + i + ' /usr/local/bin/uhd_image_loader --args=' + args)
        #os.system("omf tell -a offh -t " + i)
        time.sleep(10)    
#%% Loads images on the nodes
def load_node_image(nodes, node_image):
    print("Configuring nodes...")
    #loads the image on the node
    #os.system("omf tell -a offh -t all")
    for i, j in zip(nodes, node_image):
        print("Loading "+ j+ " image on " + i)
        os.system("omf tell -a offh -t " + i)
        os.system("omf load  -r 0 -i " + j + " -t " + i)
        os.system("omf tell -a on -t " + i)
    return 0
#%%    
def power_on_nodes(nodes):
    print("Powering on nodes")
    for i in nodes:
        os.system("omf tell -a on -t " + i)

    #Waits for each node to respond to a ping to verify that it is powered up
    for i in nodes: 
        print("Waiting for " + i + " to power on.")
        ping(i)
        time.sleep(30)
#%%    
def power_cycle_nodes(nodes):
    print("Power cycling nodes")
    for i in nodes:
        os.system("omf tell -a offh -t " + i)
        os.system("omf tell -a on -t " + i)
    #Waits for each node to respond to a ping to verify that it is powered up
    for i in nodes: 
        print("Waiting for " + i + " to power on.")
        ping(i)
#%%  
def load_node_software(nodes, sw):
    #Waits for each node to respond to a ping to verify that it is powered up
    for i in nodes:
        print()
        print("Loading software on " + i)
        os.system("ssh -o 'StrictHostKeyChecking no' root@" + i + " sudo apt update")
        for s in sw:
            os.system("ssh -o 'StrictHostKeyChecking no' root@" + i + " sudo apt install " + s)
            os.system("ssh -o 'StrictHostKeyChecking no' root@" + i + " sudo apt install " + s)

#%% Copy files to specfied destination
def copy_files(dest_in ="", dest_out = "", nodes= ["node1-2"], copy_type = "1a"):
    for node in nodes:
        print("")   
        print("Copying files for " + node)    
        #print("scp -o 'StrictHostKeyChecking no' -r " + dest_in +" root@" + node + ":" + dest_out)
        if copy_type.lower() == "1a": os.system("scp -o 'StrictHostKeyChecking no' -r root@" + node+ ":" + dest_in + " " + dest_out)
        if copy_type.lower() == "1b": os.system("scp -o 'StrictHostKeyChecking no' -r " + dest_in +" root@" + node + ":" + dest_out)
        if copy_type.lower() == "2a": os.system("ssh -o 'StrictHostKeyChecking no' root@" + node + " rclone copy -v " + dest_in + " " + dest_out)    
    return 0

#%%  
def run_cmd_in_node(nodes, command, cmd_type):
    #Runs specified command on node
    cmd = ""
    for c in command:
        cmd = cmd + " " + c
    c = " ' " + c + " ' "
    
    for i in nodes:  
        print("Running commands on " + i)
        if cmd_type[0].upper() == "P2": os.system("ssh -o 'StrictHostKeyChecking no' root@" + i + ' "python2 ' + cmd + '"')
        elif cmd_type[0].upper() == "P3": os.system("ssh -o 'StrictHostKeyChecking no' root@" + i + ' "python3 ' + cmd + '"')
        else: os.system("ssh -o 'StrictHostKeyChecking no' root@" + i + " " + cmd)
#%% Check lengths and sets to the appropriate length if necesssary. 
def check_list_length(arr_sample, arr_test):
    if len(arr_sample) == len(arr_test): arr_out = arr_test
    elif len(arr_sample) > 1 and len(arr_test) == 1: arr_out = arr_test*len(arr_sample)
    else: sys.exit("The arrays are not of the proper lengths. ")
    return arr_out
#%%
def ping(host, samples = 100):
    #Returns True if host (str) responds to a ping request.
    #Remember that a host may not respond to a ping (ICMP) request even if the host name is valid
    # Option for the number of packets as a function of
    param = '-n' if platform.system().lower()=='windows' else '-c' 
    for i in range(samples):
        # Building and sending the command. Ex: "ping -c 1 google.com"
        response = subprocess.call(['ping', param, '1', host])
        if response == 0: break
    return 

#%%Main program
def main(options = None):
    if options is None:
        options = argument_parser().parse_args()        

    def sig_handler(sig=None, frame=None):
        sys.exit(0)    
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    
    dev_addr = check_list_length(options.nodes, options.dev_addr)
    dev_type = check_list_length(dev_addr, options.dev_type)
    node_addr = check_list_length(options.nodes, options.node_addr)
    node_image = check_list_length(options.nodes, options.node_image)
    options.cmd_type = check_list_length(options.cmd, options.cmd_type)
    #com_type = check_list_length(options.command, options.cmd_type)
    
    for test in options.actions:
        if test == "load-node": load_node_image(options.nodes, node_image)
        elif test == "config-eth": conf_eth(options.nodes, node_addr)
        elif test == "power-on-node": power_on_nodes(options.nodes)
        elif test == "power-cycle-node": power_cycle_nodes(options.nodes)
        elif test == "load-dev": load_dev_image(options.nodes, dev_addr, dev_type)
        elif test == "software": load_node_software(options.nodes, options.software)
        elif test == "run-cmd": run_cmd_in_node(options.nodes, options.cmd, options.cmd_type)
        elif test == "copy-files": copy_files(options.files_test[0], options.files_copy_dest, options.nodes, options.files_copy)
        else: print("Command not known")
    return 0

#%%
if __name__ == '__main__':
    main()
   

