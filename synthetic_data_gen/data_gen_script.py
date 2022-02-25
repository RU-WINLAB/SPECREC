import data_gen
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
            "--sig2_freq", dest="sig2_freq", type=float, default=-5.0e6,
            help="Set signal 2 central frequency [default=%(default)r]")
    parser.add_argument(
            "--sig2_bw", dest="sig2_bw", type=float, default=4.0e6,
            help="Set signal 2 bandwidth [default=%(default)r]")
    parser.add_argument(
            "--run_time", dest="run_time", type=float, default=2.0,
            help="Set running time [default=%(default)r]")
    parser.add_argument(
            "--mods_list", dest="mods_list", type=str, default='qpsk,bpsk,8psk,16qam',
            help="List of all modulations [default=%(default)r]")
    return parser

def main(options=None):
    if options is None:
        options = argument_parser().parse_args()

    mods = options.mods_list.split(',')
    for i in range(len(mods)):
        for j in range(len(mods)):
            if i < j:
                args = ["--filename_base", str(options.filename), "--samp_rate", str(options.samp_rate), "--noise_voltage_range_lo", str(options.nv_start), "--noise_voltage_range_hi", str(options.nv_end), "--noise_voltage_step", str(options.nv_step), "--sig1_freq", str(options.sig1_freq), "--sig1_bw", str(options.sig1_bw), "--sig2_freq", str(options.sig2_freq), "--sig2_bw", str(options.sig2_bw), "--run_time", str(options.run_time), "--sig1_mod", mods[i], "--sig2_mod", mods[j]]
                data_gen.main(args)
                print("Data generation for modulations {} and {} for all noise voltages done!".format(mods[i], mods[j]))

if __name__ == '__main__':
    main()
