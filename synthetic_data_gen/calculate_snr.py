#!/usr/bin/env python3

import numpy as np
from argparse import ArgumentParser

def read_binary_iq(filename, **kwargs):
    data = np.fromfile(filename, dtype=np.float32, **kwargs)
    dataiq = ((data[0::2] + data[1::2]*1j).astype(np.complex))
    return dataiq

def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--filename", dest="filename", type=str, default='file1.bin',
        help="Set Filename [default=%(default)r]")
    parser.add_argument(
        "--num_sigs", dest="numsigs", type=int, default=1,
        help="Set Number of signals present [default=%(default)r]")
    return parser

def main(options=None):
    if options is None:
        options = argument_parser().parse_args()

    samples = read_binary_iq(options.filename)
    numsamps = len(samples)
    if options.numsigs == 1:
        cleansig = read_binary_iq('cleansig_' + options.filename)
        cleansig = cleansig[0:numsamps]
        noise = samples - cleansig
        snr = np.mean(abs(cleansig)) / np.mean(abs(noise))
        print ('Total analog SNR is {} dB'.format(20*np.log10(snr)))
    elif options.numsigs == 2:
        cleansig1 = read_binary_iq('cleansig1_' + options.filename)
        cleansig1 = cleansig1[0:numsamps]
        cleansig2 = read_binary_iq('cleansig2_' + options.filename)
        cleansig2 = cleansig2[0:numsamps]
        noise_interf1 = samples - cleansig1
        noise_interf2 = samples - cleansig2
        sinr1 = np.mean(abs(cleansig1)) / np.mean(abs(noise_interf1))
        sinr2 = np.mean(abs(cleansig2)) / np.mean(abs(noise_interf2))
        print ('Total analog SINR for signal 1 is {} dB'.format(20*np.log10(sinr1)))
        print ('Total analog SINR for signal 2 is {} dB'.format(20*np.log10(sinr2)))

if __name__ == '__main__':
    main()
