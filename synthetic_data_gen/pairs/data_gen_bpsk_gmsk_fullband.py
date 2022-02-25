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
from gnuradio import gr
from gnuradio.filter import firdes
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from signal_bpsk import signal_bpsk  # grc-generated hier_block
from signal_gmsk import signal_gmsk  # grc-generated hier_block

class data_gen_bpsk_gmsk_fullband(gr.top_block):

    def __init__(self, filename='data_bpsk_gmsk.bin', samp_rate=20.0e6, sig1_freq=5e6):
        gr.top_block.__init__(self, "Synthetic Data Generation")

        ##################################################
        # Parameters
        ##################################################
        self.filename = filename
        self.samp_rate = samp_rate
        self.sig1_freq = sig1_freq

        ##################################################
        # Blocks
        ##################################################
        self.signal_gmsk_0 = signal_gmsk(
            offset_freq=sig1_freq,
            samp_rate=samp_rate,
            signal_bandwidth=samp_rate,
        )
        self.signal_bpsk_0 = signal_bpsk(
            offset_freq=sig1_freq,
            samp_rate=samp_rate,
            signal_bandwidth=samp_rate,
        )
        self.blocks_throttle_0_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, filename, False)
        self.blocks_file_sink_0.set_unbuffered(False)
        self.blocks_add_xx_0 = blocks.add_vcc(1)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.blocks_throttle_0_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.signal_bpsk_0, 0), (self.blocks_throttle_0_0, 0))
        self.connect((self.signal_gmsk_0, 0), (self.blocks_throttle_0, 0))

    def get_filename(self):
        return self.filename

    def set_filename(self, filename):
        self.filename = filename
        self.blocks_file_sink_0.open(self.filename)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)
        self.blocks_throttle_0_0.set_sample_rate(self.samp_rate)
        self.signal_bpsk_0.set_samp_rate(self.samp_rate)
        self.signal_bpsk_0.set_signal_bandwidth(self.samp_rate)
        self.signal_gmsk_0.set_samp_rate(self.samp_rate)
        self.signal_gmsk_0.set_signal_bandwidth(self.samp_rate)

    def get_sig1_freq(self):
        return self.sig1_freq

    def set_sig1_freq(self, sig1_freq):
        self.sig1_freq = sig1_freq
        self.signal_bpsk_0.set_offset_freq(self.sig1_freq)
        self.signal_gmsk_0.set_offset_freq(self.sig1_freq)


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--filename", dest="filename", type=str, default='data_bpsk_gmsk.bin',
        help="Set Filename [default=%(default)r]")
    parser.add_argument(
        "--samp-rate", dest="samp_rate", type=eng_float, default="20.0M",
        help="Set Base Sampling Rate [default=%(default)r]")
    parser.add_argument(
        "--sig1-freq", dest="sig1_freq", type=eng_float, default="5.0M",
        help="Set Central Frequency of Signal 1 [default=%(default)r]")
    return parser


def main(top_block_cls=data_gen_bpsk_gmsk_fullband, options=None):
    if options is None:
        options = argument_parser().parse_args()
    tb = top_block_cls(filename=options.filename, samp_rate=options.samp_rate, sig1_freq=options.sig1_freq)

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
