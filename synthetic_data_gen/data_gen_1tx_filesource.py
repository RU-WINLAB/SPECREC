#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Generic Single Transmitter Data Gen Filesource
# Author: root
# GNU Radio version: 3.8.1.0

import os
import sys
sys.path.append(os.environ.get('GRC_HIER_PATH', os.path.expanduser('~/.grc_gnuradio')))

from gnuradio import blocks
from gnuradio import channels
from gnuradio.filter import firdes
from gnuradio import gr
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from one_tx_filesource import one_tx_filesource  # grc-generated hier_block


class data_gen_1tx_filesource(gr.top_block):

    def __init__(self, filename='file1.bin', gain=1.0, modulation='bpsk', noise_seed=0, num_samps=1000000, samp_rate=20e6, sig1_bw=4e6, sig1_freq=7e6, snr=10, sourcefile='source.bin'):
        gr.top_block.__init__(self, "Generic Single Transmitter Data Gen Filesource")

        ##################################################
        # Parameters
        ##################################################
        self.filename = filename
        self.gain = gain
        self.modulation = modulation
        self.noise_seed = noise_seed
        self.num_samps = num_samps
        self.samp_rate = samp_rate
        self.sig1_bw = sig1_bw
        self.sig1_freq = sig1_freq
        self.snr = snr
        self.sourcefile = sourcefile

        ##################################################
        # Variables
        ##################################################
        self.mod_bps = mod_bps = 1 if modulation == "bpsk" else 2 if modulation == "cpm" else 1 if modulation == "gmsk" else 2 if modulation == "qpsk" else 4 if modulation == "16qam" else 3
        self.noise_voltage = noise_voltage = 50/(10**(snr/20)) * sig1_bw * mod_bps/samp_rate

        ##################################################
        # Blocks
        ##################################################
        self.one_tx_filesource_0 = one_tx_filesource(
            modulation=modulation,
            samp_rate=samp_rate,
            sig_bw=sig1_bw,
            sig_freq=sig1_freq,
            sourcefile=sourcefile,
        )
        self.channels_channel_model_0 = channels.channel_model(
            noise_voltage=noise_voltage,
            frequency_offset=0.0,
            epsilon=1.0,
            taps=[1.0 + 1.0j],
            noise_seed=noise_seed,
            block_tags=False)
        self.blocks_throttle_0_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(gain)
        self.blocks_head_1 = blocks.head(gr.sizeof_gr_complex*1, num_samps)
        self.blocks_file_sink_1 = blocks.file_sink(gr.sizeof_gr_complex*1, 'cleansig_' + filename, False)
        self.blocks_file_sink_1.set_unbuffered(False)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, filename, False)
        self.blocks_file_sink_0.set_unbuffered(False)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_head_1, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.blocks_throttle_0_0, 0))
        self.connect((self.blocks_throttle_0_0, 0), (self.blocks_file_sink_1, 0))
        self.connect((self.blocks_throttle_0_0, 0), (self.channels_channel_model_0, 0))
        self.connect((self.channels_channel_model_0, 0), (self.blocks_head_1, 0))
        self.connect((self.one_tx_filesource_0, 0), (self.blocks_multiply_const_vxx_0, 0))


    def get_filename(self):
        return self.filename

    def set_filename(self, filename):
        self.filename = filename
        self.blocks_file_sink_0.open(self.filename)
        self.blocks_file_sink_1.open('cleansig_' + self.filename)

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.blocks_multiply_const_vxx_0.set_k(self.gain)

    def get_modulation(self):
        return self.modulation

    def set_modulation(self, modulation):
        self.modulation = modulation
        self.set_mod_bps(1 if self.modulation == "bpsk" else 2 if self.modulation == "cpm" else 1 if self.modulation == "gmsk" else 2 if self.modulation == "qpsk" else 4 if self.modulation == "16qam" else 3)
        self.one_tx_filesource_0.set_modulation(self.modulation)

    def get_noise_seed(self):
        return self.noise_seed

    def set_noise_seed(self, noise_seed):
        self.noise_seed = noise_seed

    def get_num_samps(self):
        return self.num_samps

    def set_num_samps(self, num_samps):
        self.num_samps = num_samps
        self.blocks_head_1.set_length(self.num_samps)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_noise_voltage(50/(10**(self.snr/20)) * self.sig1_bw * self.mod_bps/self.samp_rate)
        self.blocks_throttle_0_0.set_sample_rate(self.samp_rate)
        self.one_tx_filesource_0.set_samp_rate(self.samp_rate)

    def get_sig1_bw(self):
        return self.sig1_bw

    def set_sig1_bw(self, sig1_bw):
        self.sig1_bw = sig1_bw
        self.set_noise_voltage(50/(10**(self.snr/20)) * self.sig1_bw * self.mod_bps/self.samp_rate)
        self.one_tx_filesource_0.set_sig_bw(self.sig1_bw)

    def get_sig1_freq(self):
        return self.sig1_freq

    def set_sig1_freq(self, sig1_freq):
        self.sig1_freq = sig1_freq
        self.one_tx_filesource_0.set_sig_freq(self.sig1_freq)

    def get_snr(self):
        return self.snr

    def set_snr(self, snr):
        self.snr = snr
        self.set_noise_voltage(50/(10**(self.snr/20)) * self.sig1_bw * self.mod_bps/self.samp_rate)

    def get_sourcefile(self):
        return self.sourcefile

    def set_sourcefile(self, sourcefile):
        self.sourcefile = sourcefile
        self.one_tx_filesource_0.set_sourcefile(self.sourcefile)

    def get_mod_bps(self):
        return self.mod_bps

    def set_mod_bps(self, mod_bps):
        self.mod_bps = mod_bps
        self.set_noise_voltage(50/(10**(self.snr/20)) * self.sig1_bw * self.mod_bps/self.samp_rate)

    def get_noise_voltage(self):
        return self.noise_voltage

    def set_noise_voltage(self, noise_voltage):
        self.noise_voltage = noise_voltage
        self.channels_channel_model_0.set_noise_voltage(self.noise_voltage)




def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--filename", dest="filename", type=str, default='file1.bin',
        help="Set Filename [default=%(default)r]")
    parser.add_argument(
        "--gain", dest="gain", type=eng_float, default="1.0",
        help="Set Gain (use value lower than 1 for attenuation) [default=%(default)r]")
    parser.add_argument(
        "--modulation", dest="modulation", type=str, default='bpsk',
        help="Set Modulation type ('bpsk', 'cpm', 'gmsk', 'qpsk', '16qam', 8psk') [default=%(default)r]")
    parser.add_argument(
        "--noise-seed", dest="noise_seed", type=intx, default=0,
        help="Set Noise seed [default=%(default)r]")
    parser.add_argument(
        "--num-samps", dest="num_samps", type=intx, default=1000000,
        help="Set Number of samples [default=%(default)r]")
    parser.add_argument(
        "--samp-rate", dest="samp_rate", type=eng_float, default="20.0M",
        help="Set Base Sampling Rate [default=%(default)r]")
    parser.add_argument(
        "--sig1-bw", dest="sig1_bw", type=eng_float, default="4.0M",
        help="Set Bandwidth of the Signal [default=%(default)r]")
    parser.add_argument(
        "--sig1-freq", dest="sig1_freq", type=eng_float, default="7.0M",
        help="Set Central Frequency of the Signal [default=%(default)r]")
    parser.add_argument(
        "--snr", dest="snr", type=eng_float, default="10.0",
        help="Set Signal to Noise Ratio [default=%(default)r]")
    parser.add_argument(
        "--sourcefile", dest="sourcefile", type=str, default='source.bin',
        help="Set Source Filename [default=%(default)r]")
    return parser


def main(top_block_cls=data_gen_1tx_filesource, options=None):
    if options is None:
        options = argument_parser().parse_args()
    tb = top_block_cls(filename=options.filename, gain=options.gain, modulation=options.modulation, noise_seed=options.noise_seed, num_samps=options.num_samps, samp_rate=options.samp_rate, sig1_bw=options.sig1_bw, sig1_freq=options.sig1_freq, snr=options.snr, sourcefile=options.sourcefile)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    tb.wait()


if __name__ == '__main__':
    main()
