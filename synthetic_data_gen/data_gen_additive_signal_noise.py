#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Single Transmitter Data Gen With Additive Noise
# Author: root
# GNU Radio version: 3.8.0.0

from distutils.version import StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

import os
import sys
sys.path.append(os.environ.get('GRC_HIER_PATH', os.path.expanduser('~/.grc_gnuradio')))

from PyQt5 import Qt
from gnuradio import qtgui
import sip
from gnuradio import analog
from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from one_tx import one_tx  # grc-generated hier_block
from gnuradio import qtgui

class data_gen_additive_signal_noise(gr.top_block, Qt.QWidget):

    def __init__(self, filename='file1.bin', modulation='bpsk', noise_gain=1.0, num_samps=10000, samp_rate=10e3, sig1_bw=10000, sig1_freq=1000, signal_gain=1.0, snr=100):
        gr.top_block.__init__(self, "Single Transmitter Data Gen With Additive Noise")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Single Transmitter Data Gen With Additive Noise")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "data_gen_additive_signal_noise")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Parameters
        ##################################################
        self.filename = filename
        self.modulation = modulation
        self.noise_gain = noise_gain
        self.num_samps = num_samps
        self.samp_rate = samp_rate
        self.sig1_bw = sig1_bw
        self.sig1_freq = sig1_freq
        self.signal_gain = signal_gain
        self.snr = snr

        ##################################################
        # Blocks
        ##################################################
        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(
            1024, #size
            "", #name
            1 #number of inputs
        )
        self.qtgui_const_sink_x_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0.enable_grid(False)
        self.qtgui_const_sink_x_0.enable_axis_labels(True)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
            "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0.pyqwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_const_sink_x_0_win)
        self.one_tx_0 = one_tx(
            modulation=modulation,
            samp_rate=samp_rate,
            sig_bw=sig1_bw,
            sig_freq=sig1_freq,
        )
        self.blocks_throttle_0_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_multiply_const_vxx_0_0 = blocks.multiply_const_cc(noise_gain)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(signal_gain)
        self.blocks_head_1 = blocks.head(gr.sizeof_gr_complex*1, num_samps)
        self.blocks_file_sink_1 = blocks.file_sink(gr.sizeof_gr_complex*1, 'cleansig_' + filename, False)
        self.blocks_file_sink_1.set_unbuffered(False)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, filename, False)
        self.blocks_file_sink_0.set_unbuffered(False)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_UNIFORM, 1, 0)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_multiply_const_vxx_0_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_head_1, 0))
        self.connect((self.blocks_head_1, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.blocks_throttle_0_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.blocks_throttle_0_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.blocks_throttle_0_0, 0), (self.blocks_file_sink_1, 0))
        self.connect((self.one_tx_0, 0), (self.blocks_multiply_const_vxx_0, 0))

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "data_gen_additive_signal_noise")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_filename(self):
        return self.filename

    def set_filename(self, filename):
        self.filename = filename
        self.blocks_file_sink_0.open(self.filename)
        self.blocks_file_sink_1.open('cleansig_' + self.filename)

    def get_modulation(self):
        return self.modulation

    def set_modulation(self, modulation):
        self.modulation = modulation
        self.one_tx_0.set_modulation(self.modulation)

    def get_noise_gain(self):
        return self.noise_gain

    def set_noise_gain(self, noise_gain):
        self.noise_gain = noise_gain
        self.blocks_multiply_const_vxx_0_0.set_k(self.noise_gain)

    def get_num_samps(self):
        return self.num_samps

    def set_num_samps(self, num_samps):
        self.num_samps = num_samps
        self.blocks_head_1.set_length(self.num_samps)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0_0.set_sample_rate(self.samp_rate)
        self.one_tx_0.set_samp_rate(self.samp_rate)

    def get_sig1_bw(self):
        return self.sig1_bw

    def set_sig1_bw(self, sig1_bw):
        self.sig1_bw = sig1_bw
        self.one_tx_0.set_sig_bw(self.sig1_bw)

    def get_sig1_freq(self):
        return self.sig1_freq

    def set_sig1_freq(self, sig1_freq):
        self.sig1_freq = sig1_freq
        self.one_tx_0.set_sig_freq(self.sig1_freq)

    def get_signal_gain(self):
        return self.signal_gain

    def set_signal_gain(self, signal_gain):
        self.signal_gain = signal_gain
        self.blocks_multiply_const_vxx_0.set_k(self.signal_gain)

    def get_snr(self):
        return self.snr

    def set_snr(self, snr):
        self.snr = snr


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--filename", dest="filename", type=str, default='file1.bin',
        help="Set Filename [default=%(default)r]")
    parser.add_argument(
        "--modulation", dest="modulation", type=str, default='bpsk',
        help="Set Modulation type ('bpsk', 'cpm', 'gmsk', 'qpsk', '16qam', 8psk') [default=%(default)r]")
    parser.add_argument(
        "--noise-gain", dest="noise_gain", type=eng_float, default="1.0",
        help="Set Noise gain [default=%(default)r]")
    parser.add_argument(
        "--num-samps", dest="num_samps", type=intx, default=10000,
        help="Set Number of samples [default=%(default)r]")
    parser.add_argument(
        "--samp-rate", dest="samp_rate", type=eng_float, default="10.0k",
        help="Set Base Sampling Rate [default=%(default)r]")
    parser.add_argument(
        "--sig1-bw", dest="sig1_bw", type=eng_float, default="10.0k",
        help="Set Bandwidth of the Signal [default=%(default)r]")
    parser.add_argument(
        "--sig1-freq", dest="sig1_freq", type=eng_float, default="1.0k",
        help="Set Central Frequency of the Signal [default=%(default)r]")
    parser.add_argument(
        "--signal-gain", dest="signal_gain", type=eng_float, default="1.0",
        help="Set Signal gain (use value lower than 1 for attenuation) [default=%(default)r]")
    parser.add_argument(
        "--snr", dest="snr", type=eng_float, default="100.0",
        help="Set Signal to Noise Ratio [default=%(default)r]")
    return parser


def main(top_block_cls=data_gen_additive_signal_noise, options=None):
    if options is None:
        options = argument_parser().parse_args()

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(filename=options.filename, modulation=options.modulation, noise_gain=options.noise_gain, num_samps=options.num_samps, samp_rate=options.samp_rate, sig1_bw=options.sig1_bw, sig1_freq=options.sig1_freq, signal_gain=options.signal_gain, snr=options.snr)
    tb.start()
    tb.show()

    def sig_handler(sig=None, frame=None):
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    def quitting():
        tb.stop()
        tb.wait()
    qapp.aboutToQuit.connect(quitting)
    qapp.exec_()


if __name__ == '__main__':
    main()
