from __future__ import division, print_function

import os
from shutil import rmtree
import argparse
import numpy as np
import matplotlib      # Remove this line if you don't need them
matplotlib.use('Agg')  # Remove this line if you don't need them
import matplotlib.pyplot as plt

import soundfile as sf
# import librosa
import pyworld as pw
from pitch2freq import *

song_dict = {}

EPSILON = 1e-8

def get_args():
    parser = argparse.ArgumentParser(description='Argument Parser for speech to singing synthesis')
    parser.add_argument('-i', '--input', type=str, default="./utterance/1.k.wav", required=False)
    parser.add_argument('-o', '--output', type=str, default="./test/output.wav", required=False)
    parser.add_argument("-f", "--frame_period", type=float, default=5.0)
    parser.add_argument("-s", "--speed", type=int, default=10)
    # parser.add_argument('-n', '--note', required=True)
    args = parser.parse_args()
    return args

def savefig(filename, figlist, log=True):
    #h = 10
    n = len(figlist)
    # peek into instances
    f = figlist[0]
    if len(f.shape) == 1:
        plt.figure()
        for i, f in enumerate(figlist):
            plt.subplot(n, 1, i+1)
            if len(f.shape) == 1:
                plt.plot(f)
                plt.xlim([0, len(f)])
    elif len(f.shape) == 2:
        Nsmp, dim = figlist[0].shape
        #figsize=(h * float(Nsmp) / dim, len(figlist) * h)
        #plt.figure(figsize=figsize)
        plt.figure()
        for i, f in enumerate(figlist):
            plt.subplot(n, 1, i+1)
            if log:
                x = np.log(f + EPSILON)
            else:
                x = f + EPSILON
            plt.imshow(x.T, origin='lower', interpolation='none', aspect='auto', extent=(0, x.shape[0], 0, x.shape[1]))
    else:
        raise ValueError('Input dimension must < 3.')
    plt.savefig(filename)
    # plt.close()

def calculate_f0(f0_o, note):
    std = np.zeros(f0_o.shape)
    sum_f0 = 0
    num_f0 = 0
    # calculate average f0
    for f in f0_o:
        if f != 0:
            num_f0 += 1
            sum_f0 += f
    avg_f0 = sum_f0 / num_f0

    # print(f"Average f0: {avg_f0}, {len(f0_o) - num_f0} 0 in f0")

    for idx, f in enumerate(f0_o):
        if f == 0:
            std[idx] = 0
        else:
            std[idx] = f - avg_f0
    
    pitch = pitch2freq(note)
    new_f0 = np.zeros(f0_o.shape)
    for idx, s in enumerate(std):
        if s == 0:
            new_f0[idx] = 0
        else:
            new_f0[idx] = pitch + s

    return new_f0

def make_song_dict(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            f_name = f.split('.')[0]
            song_dict[f_name] = path + f

# def test(note_number, note_list):
#     for idx, note in 




def main(args):
    if not os.path.isdir('test'):
        os.mkdir('test')
        # rmtree('test')
    # os.mkdir('test')

    # args = get_args()

    notes_number = int(input("How many note are you going to compose?"))
    print()
    note_list = list(str(num) for num in input("Enter the note items separated by space: ").strip().split())[:notes_number]
    note_len = len(note_list)
    print(note_list, note_len)

    speech = args.input
    x, fs = sf.read(speech)
    # x, fs = librosa.load('utterance/vaiueo2d.wav', dtype=np.float64)

    # 1. A convient way
    f0, sp, ap = pw.wav2world(x, fs)    # use default options # numpy array
    y = pw.synthesize(f0, sp, ap, fs, pw.default_frame_period)

    # 2. Step by step
    # 2-1 Without F0 refinement
    _f0, t = pw.dio(x, fs, f0_floor=50.0, f0_ceil=600.0,
                    channels_in_octave=2,
                    frame_period=args.frame_period,
                    speed=args.speed)
    _sp = pw.cheaptrick(x, _f0, t, fs)
    _ap = pw.d4c(x, _f0, t, fs)
    _y = pw.synthesize(_f0, _sp, _ap, fs, args.frame_period)
    print("_f0: ", _f0)
    print("shape(_f0): ", _f0.shape)
    std = np.zeros(_f0.shape)
    num_f0 = 0
    sum_f0 = 0
    for f in _f0:
        if f != 0:
            num_f0 += 1
            sum_f0 += f
    print("num_f0: ", num_f0)
    print("Average _f0: ", sum_f0 / num_f0)
    # librosa.output.write_wav('test/y_without_f0_refinement.wav', _y, fs)
    sf.write('test/y_without_f0_refinement.wav', _y, fs)
    print(_y)
    print(_y.shape)


    # 2-2 DIO with F0 refinement (using Stonemask)
    f0 = pw.stonemask(x, _f0, t, fs)
    sp = pw.cheaptrick(x, f0, t, fs)
    ap = pw.d4c(x, f0, t, fs)
    y = pw.synthesize(f0, sp, ap, fs, args.frame_period)
    # librosa.output.write_wav('test/y_with_f0_refinement.wav', y, fs)
    sf.write('test/y_with_f0_refinement.wav', y, fs)
    print("f0: ", f0)
    print("shape(f0): ", f0.shape)
    num_f0 = 0
    sum_f0 = 0
    for f in f0:
        if f != 0:
            num_f0 += 1
            sum_f0 += f
    print("num_f0: ", num_f0)
    print("Average f0: ", sum_f0 / num_f0)
    print(y.shape)
    print(_y.shape)


    # 2-3 Harvest with F0 refinement (using Stonemask)
    _f0_h, t_h = pw.harvest(x, fs)
    f0_h = pw.stonemask(x, _f0_h, t_h, fs)
    sp_h = pw.cheaptrick(x, f0_h, t_h, fs)
    ap_h = pw.d4c(x, f0_h, t_h, fs)
    y_h = pw.synthesize(f0_h, sp_h, ap_h, fs, pw.default_frame_period)
    # librosa.output.write_wav('test/y_harvest_with_f0_refinement.wav', y_h, fs)
    sf.write('test/y_harvest_with_f0_refinement.wav', y_h, fs)


    all_y = np.concatenate((_y, y, y_h))
    sf.write('test/y.wav', all_y, fs)

    ### Testing Area 
    all_y = np.zeros(_y.shape)
    for jdx in range(notes_number):
        new_f0 = calculate_f0(f0_h, note_list[jdx])
        sp_h = pw.cheaptrick(x, new_f0, t_h, fs)
        ap_h = pw.d4c(x, new_f0, t_h, fs)
        new_y = pw.synthesize(new_f0, sp_h, ap_h, fs, pw.default_frame_period)
        if jdx == 0:
            all_y = new_y
        else:
            all_y = np.concatenate((all_y, new_y))
    sf.write(args.output, all_y, fs)

    # Another test
    make_song_dict("./perfect_cut/")

    ####




    # Comparison
    savefig('test/wavform.png', [x, _y, y])
    savefig('test/sp.png', [_sp, sp])
    savefig('test/ap.png', [_ap, ap], log=False)
    savefig('test/f0.png', [_f0, f0])

    print('Please check "test" directory for output files')


if __name__ == '__main__':
    args = get_args()
    main(args)

# C3 G3 E3 D3 C3
