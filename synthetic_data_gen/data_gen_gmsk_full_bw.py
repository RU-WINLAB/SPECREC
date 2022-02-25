#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: GMSK full bandwidth
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
from signal_gmsk import signal_gmsk  # grc-generated hier_block

class data_gen_gmsk_full_bw(gr.top_block):

    def __init__(self, samp_rate=2.0e6):
        gr.top_block.__init__(self, "GMSK full bandwidth")

        ##################################################
        # Parameters
        ##################################################
        self.samp_rate = samp_rate

        ##################################################
        # Variables
        ##################################################
        self.sig1_bw = sig1_bw = samp_rate

        ##################################################
        # Blocks
        ##################################################
        self.signal_gmsk_0 = signal_gmsk(
            offset_freq=0,
            samp_rate=samp_rate,
            signal_bandwidth=sig1_bw,
        )
        self.blocks_throttle_0_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, 'gmsk_full_bw.bin', False)
        self.blocks_file_sink_0.set_unbuffered(False)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_throttle_0_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.signal_gmsk_0, 0), (self.blocks_throttle_0_0, 0))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_sig1_bw(self.samp_rate)
        self.blocks_throttle_0_0.set_sample_rate(self.samp_rate)
        self.signal_gmsk_0.set_samp_rate(self.samp_rate)

    def get_sig1_bw(self):
        return self.sig1_bw

    def set_sig1_bw(self, sig1_bw):
        self.sig1_bw = sig1_bw
        self.signal_gmsk_0.set_signal_bandwidth(self.sig1_bw)


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--samp-rate", dest="samp_rate", type=eng_float, default="2.0M",
        help="Set Base Sampling Rate [default=%(default)r]")
    return parser


def main(top_block_cls=data_gen_gmsk_full_bw, options=None):
    if options is None:
        options = argument_parser().parse_args()
    tb = top_block_cls(samp_rate=options.samp_rate)

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
