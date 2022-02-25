#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: phase_gen
# Author: tld95
# Description: This module will be used to generate a contant phase shifft in one direction
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
from PyQt5.QtCore import QObject, pyqtSlot
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import analog
from gnuradio import blocks
from gnuradio import gr
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio.qtgui import Range, RangeWidget
from one_tx import one_tx  # grc-generated hier_block
from gnuradio import qtgui

class phase_gen(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "phase_gen")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("phase_gen")
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

        self.settings = Qt.QSettings("GNU Radio", "phase_gen")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.wave_freq = wave_freq = 1000
        self.var_sine_freq_adj = var_sine_freq_adj = .5
        self.var_phase_amp_adj = var_phase_amp_adj = .5
        self.phase_amp = phase_amp = .1
        self.mod_selector = mod_selector = 'bpsk'
        self.freq_adj = freq_adj = 1000

        ##################################################
        # Blocks
        ##################################################
        self._var_sine_freq_adj_range = Range(0, 1, .001, .5, 200)
        self._var_sine_freq_adj_win = RangeWidget(self._var_sine_freq_adj_range, self.set_var_sine_freq_adj, 'var_sine_freq_adj', "counter_slider", float)
        self.top_grid_layout.addWidget(self._var_sine_freq_adj_win)
        self._var_phase_amp_adj_range = Range(0, 10, .01, .5, 200)
        self._var_phase_amp_adj_win = RangeWidget(self._var_phase_amp_adj_range, self.set_var_phase_amp_adj, 'var_phase_amp_adj', "counter_slider", float)
        self.top_grid_layout.addWidget(self._var_phase_amp_adj_win)
        # Create the options list
        self._mod_selector_options = ('bpsk', 'qpsk', '8psk', '16qam', )
        # Create the labels list
        self._mod_selector_labels = ('bpsk', 'qpsk', '8psk', '16qam', )
        # Create the combo box
        self._mod_selector_tool_bar = Qt.QToolBar(self)
        self._mod_selector_tool_bar.addWidget(Qt.QLabel(mod_selector + ": "))
        self._mod_selector_combo_box = Qt.QComboBox()
        self._mod_selector_tool_bar.addWidget(self._mod_selector_combo_box)
        for _label in self._mod_selector_labels: self._mod_selector_combo_box.addItem(_label)
        self._mod_selector_callback = lambda i: Qt.QMetaObject.invokeMethod(self._mod_selector_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._mod_selector_options.index(i)))
        self._mod_selector_callback(self.mod_selector)
        self._mod_selector_combo_box.currentIndexChanged.connect(
            lambda i: self.set_mod_selector(self._mod_selector_options[i]))
        # Create the radio buttons
        self.top_grid_layout.addWidget(self._mod_selector_tool_bar)
        self._freq_adj_range = Range(0, 1000000, 1, 1000, 200)
        self._freq_adj_win = RangeWidget(self._freq_adj_range, self.set_freq_adj, 'freq_adj', "counter", float)
        self.top_grid_layout.addWidget(self._freq_adj_win)
        self.qtgui_sink_x_0 = qtgui.sink_c(
            1024, #fftsize
            firdes.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            100, #bw
            "", #name
            True, #plotfreq
            True, #plotwaterfall
            True, #plottime
            True #plotconst
        )
        self.qtgui_sink_x_0.set_update_time(1.0/10)
        self._qtgui_sink_x_0_win = sip.wrapinstance(self.qtgui_sink_x_0.pyqwidget(), Qt.QWidget)

        self.qtgui_sink_x_0.enable_rf_freq(False)

        self.top_grid_layout.addWidget(self._qtgui_sink_x_0_win)
        self.one_tx_0 = one_tx(
            modulation=mod_selector,
            samp_rate=freq_adj,
            sig_bw=freq_adj,
            sig_freq=7e6,
        )
        self.blocks_transcendental_0_0 = blocks.transcendental('sin', "float")
        self.blocks_transcendental_0 = blocks.transcendental('cos', "float")
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_ff(var_phase_amp_adj)
        self.blocks_float_to_complex_0 = blocks.float_to_complex(1)
        self.blocks_add_const_vxx_0 = blocks.add_const_ff(1)
        self.analog_sig_source_x_0 = analog.sig_source_f(1000, analog.GR_SAW_WAVE, var_sine_freq_adj, 1, 0, 0)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_add_const_vxx_0, 0))
        self.connect((self.blocks_add_const_vxx_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.blocks_float_to_complex_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.blocks_transcendental_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.blocks_transcendental_0_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.qtgui_sink_x_0, 0))
        self.connect((self.blocks_transcendental_0, 0), (self.blocks_float_to_complex_0, 0))
        self.connect((self.blocks_transcendental_0_0, 0), (self.blocks_float_to_complex_0, 1))
        self.connect((self.one_tx_0, 0), (self.blocks_multiply_xx_0, 0))

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "phase_gen")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_wave_freq(self):
        return self.wave_freq

    def set_wave_freq(self, wave_freq):
        self.wave_freq = wave_freq

    def get_var_sine_freq_adj(self):
        return self.var_sine_freq_adj

    def set_var_sine_freq_adj(self, var_sine_freq_adj):
        self.var_sine_freq_adj = var_sine_freq_adj
        self.analog_sig_source_x_0.set_frequency(self.var_sine_freq_adj)

    def get_var_phase_amp_adj(self):
        return self.var_phase_amp_adj

    def set_var_phase_amp_adj(self, var_phase_amp_adj):
        self.var_phase_amp_adj = var_phase_amp_adj
        self.blocks_multiply_const_vxx_0.set_k(self.var_phase_amp_adj)

    def get_phase_amp(self):
        return self.phase_amp

    def set_phase_amp(self, phase_amp):
        self.phase_amp = phase_amp

    def get_mod_selector(self):
        return self.mod_selector

    def set_mod_selector(self, mod_selector):
        self.mod_selector = mod_selector
        self._mod_selector_callback(self.mod_selector)
        self.one_tx_0.set_modulation(self.mod_selector)

    def get_freq_adj(self):
        return self.freq_adj

    def set_freq_adj(self, freq_adj):
        self.freq_adj = freq_adj
        self.one_tx_0.set_samp_rate(self.freq_adj)
        self.one_tx_0.set_sig_bw(self.freq_adj)



def main(top_block_cls=phase_gen, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()
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
