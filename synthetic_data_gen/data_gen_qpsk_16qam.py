#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Synthetic Data Generation
# Author: root
# GNU Radio version: 3.8.0.0

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
from signal_16qam import signal_16qam  # grc-generated hier_block
from signal_QPSK import signal_QPSK  # grc-generated hier_block

class data_gen_qpsk_16qam(gr.top_block):

    def __init__(self, filename='file1.bin', noise_voltage=0.150, samp_rate=20e6, sig1_bw=1.5e6, sig1_freq=7e6, sig2_bw=4e6, sig2_freq=-5e6):
        gr.top_block.__init__(self, "Synthetic Data Generation")

        ##################################################
        # Parameters
        ##################################################
        self.filename = filename
        self.noise_voltage = noise_voltage
        self.samp_rate = samp_rate
        self.sig1_bw = sig1_bw
        self.sig1_freq = sig1_freq
        self.sig2_bw = sig2_bw
        self.sig2_freq = sig2_freq

        ##################################################
        # Blocks
        ##################################################
        self.signal_QPSK_0 = signal_QPSK(
            offset_freq=sig1_freq,
            samp_rate=samp_rate,
            signal_bandwidth=sig1_bw,
        )
        self.signal_16qam_0 = signal_16qam(
            offset_freq=sig2_freq,
            samp_rate=samp_rate,
            signal_bandwidth=sig2_bw,
        )
        self.channels_channel_model_0 = channels.channel_model(
            noise_voltage=noise_voltage,
            frequency_offset=0.0,
            epsilon=1.0,
            taps=[1.0 + 1.0j],
            noise_seed=0,
            block_tags=False)
        self.blocks_throttle_0_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, filename, False)
        self.blocks_file_sink_0.set_unbuffered(False)
        self.blocks_add_xx_0 = blocks.add_vcc(1)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_add_xx_0, 0), (self.channels_channel_model_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.blocks_throttle_0_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.channels_channel_model_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.signal_16qam_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.signal_QPSK_0, 0), (self.blocks_throttle_0_0, 0))

    def get_filename(self):
        return self.filename

    def set_filename(self, filename):
        self.filename = filename
        self.blocks_file_sink_0.open(self.filename)

    def get_noise_voltage(self):
        return self.noise_voltage

    def set_noise_voltage(self, noise_voltage):
        self.noise_voltage = noise_voltage
        self.channels_channel_model_0.set_noise_voltage(self.noise_voltage)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)
        self.blocks_throttle_0_0.set_sample_rate(self.samp_rate)
        self.signal_16qam_0.set_samp_rate(self.samp_rate)
        self.signal_QPSK_0.set_samp_rate(self.samp_rate)

    def get_sig1_bw(self):
        return self.sig1_bw

    def set_sig1_bw(self, sig1_bw):
        self.sig1_bw = sig1_bw
        self.signal_QPSK_0.set_signal_bandwidth(self.sig1_bw)

    def get_sig1_freq(self):
        return self.sig1_freq

    def set_sig1_freq(self, sig1_freq):
        self.sig1_freq = sig1_freq
        self.signal_QPSK_0.set_offset_freq(self.sig1_freq)

    def get_sig2_bw(self):
        return self.sig2_bw

    def set_sig2_bw(self, sig2_bw):
        self.sig2_bw = sig2_bw
        self.signal_16qam_0.set_signal_bandwidth(self.sig2_bw)

    def get_sig2_freq(self):
        return self.sig2_freq

    def set_sig2_freq(self, sig2_freq):
        self.sig2_freq = sig2_freq
        self.signal_16qam_0.set_offset_freq(self.sig2_freq)


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--filename", dest="filename", type=str, default='file1.bin',
        help="Set Filename [default=%(default)r]")
    parser.add_argument(
        "--noise-voltage", dest="noise_voltage", type=eng_float, default="150.0m",
        help="Set Noise Voltage [default=%(default)r]")
    parser.add_argument(
        "--samp-rate", dest="samp_rate", type=eng_float, default="20.0M",
        help="Set Base Sampling Rate [default=%(default)r]")
    parser.add_argument(
        "--sig1-bw", dest="sig1_bw", type=eng_float, default="1.5M",
        help="Set Bandwidth of Signal 1 [default=%(default)r]")
    parser.add_argument(
        "--sig1-freq", dest="sig1_freq", type=eng_float, default="7.0M",
        help="Set Central Frequency of Signal 1 [default=%(default)r]")
    parser.add_argument(
        "--sig2-bw", dest="sig2_bw", type=eng_float, default="4.0M",
        help="Set Bandwidth of Signal 2 [default=%(default)r]")
    parser.add_argument(
        "--sig2-freq", dest="sig2_freq", type=eng_float, default="-5.0M",
        help="Set Central Frequency of Signal 2 [default=%(default)r]")
    return parser


def main(top_block_cls=data_gen_qpsk_16qam, options=None):
    if options is None:
        options = argument_parser().parse_args()
    tb = top_block_cls(filename=options.filename, noise_voltage=options.noise_voltage, samp_rate=options.samp_rate, sig1_bw=options.sig1_bw, sig1_freq=options.sig1_freq, sig2_bw=options.sig2_bw, sig2_freq=options.sig2_freq)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
