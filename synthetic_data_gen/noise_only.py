#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Noise source only
# GNU Radio version: 3.8.0.0

from gnuradio import analog
from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation

class noise_only(gr.top_block):

    def __init__(self, amplitude=0.7, filename='filename.bin', num_samples=100000, seed=0):
        gr.top_block.__init__(self, "Noise source only")

        ##################################################
        # Parameters
        ##################################################
        self.amplitude = amplitude
        self.filename = filename
        self.num_samples = num_samples
        self.seed = seed

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 32000

        ##################################################
        # Blocks
        ##################################################
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, num_samples)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, filename, False)
        self.blocks_file_sink_0.set_unbuffered(False)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_UNIFORM, amplitude, seed)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_head_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_head_0, 0))

    def get_amplitude(self):
        return self.amplitude

    def set_amplitude(self, amplitude):
        self.amplitude = amplitude
        self.analog_noise_source_x_0.set_amplitude(self.amplitude)

    def get_filename(self):
        return self.filename

    def set_filename(self, filename):
        self.filename = filename
        self.blocks_file_sink_0.open(self.filename)

    def get_num_samples(self):
        return self.num_samples

    def set_num_samples(self, num_samples):
        self.num_samples = num_samples
        self.blocks_head_0.set_length(self.num_samples)

    def get_seed(self):
        return self.seed

    def set_seed(self, seed):
        self.seed = seed

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--filename", dest="filename", type=str, default='filename.bin',
        help="Set File name [default=%(default)r]")
    return parser


def main(top_block_cls=noise_only, options=None):
    if options is None:
        options = argument_parser().parse_args()
    tb = top_block_cls(filename=options.filename)

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
