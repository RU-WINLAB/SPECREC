#!/usr/bin/env python3

import numpy as np
import time
import subprocess
from argparse import ArgumentParser

def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
            "--filename_base", dest="filename", type=str, default="samples",
            help="Set filename [default=%(default)r]")
    parser.add_argument(
            "--samp_rate", dest="samp_rate", type=float, default=20.0e6,
            help="Set base bandwidth [default=%(default)r]")
    parser.add_argument(
            "--noise_voltage_range_lo", dest="nv_start", type=float, default=0.1,
            help="Set noise voltage start [default=%(default)r]")
    parser.add_argument(
            "--noise_voltage_range_hi", dest="nv_end", type=float, default=1.01,
            help="Set noise voltage end [default=%(default)r]")
    parser.add_argument(
            "--noise_voltage_step", dest="nv_step", type=float, default=0.1,
            help="Set noise voltage step [default=%(default)r]")
    parser.add_argument(
            "--sig1_freq", dest="sig1_freq", type=float, default=7.0e6,
            help="Set signal 1 central frequency [default=%(default)r]")
    parser.add_argument(
            "--sig1_bw", dest="sig1_bw", type=float, default=1.5e6,
            help="Set signal 1 bandwidth [default=%(default)r]")
    parser.add_argument(
            "--sig1_mod", dest="sig1_mod", type=str, default="bpsk",
            help="Set signal 1 modulation ('bpsk', 'qpsk', '8psk', '16qam') [default=%(default)r]")
    parser.add_argument(
            "--sig2_freq", dest="sig2_freq", type=float, default=-5.0e6,
            help="Set signal 2 central frequency [default=%(default)r]")
    parser.add_argument(
            "--sig2_bw", dest="sig2_bw", type=float, default=4.0e6,
            help="Set signal 2 bandwidth [default=%(default)r]")
    parser.add_argument(
            "--sig2_mod", dest="sig2_mod", type=str, default="qpsk",
            help="Set signal 2 modulation ('bpsk', 'qpsk', '8psk', '16qam') [default=%(default)r]")
    parser.add_argument(
            "--run_time", dest="run_time", type=float, default=2.0,
            help="Set running time [default=%(default)r]")
    return parser

def main(args):
    options = argument_parser().parse_args(args)

    for nv in np.arange(options.nv_start, options.nv_end, options.nv_step):
        args = ["./data_gen_" + options.sig1_mod + "_" + options.sig2_mod + ".py", "--filename", options.filename + "_" + options.sig1_mod + "_" + options.sig2_mod + "noise_" + str(nv) + ".bin", "--samp-rate", str(options.samp_rate), "--noise-voltage", str(nv), "--sig1-bw", str(options.sig1_bw), "--sig1-freq", str(options.sig1_freq), "--sig2-bw", str(options.sig2_bw), "--sig2-freq", str(options.sig2_freq)]
        proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        time.sleep(options.run_time)
        proc.communicate(input=b'\n')
        print("Data generation for modulations {} and {}, noise level {}, done!".format(options.sig1_mod, options.sig2_mod, nv))

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
