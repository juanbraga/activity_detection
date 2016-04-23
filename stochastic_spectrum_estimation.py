# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 21:05:21 2016

@author: jbraga
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav


if __name__ == "__main__":  

    import csv
    dataset=[]    
    cr = csv.reader(open('dataset.csv',"rb"))
    for row in cr:
        dataset.append(row[0]) 
    
    audio_file = dataset[0] + '_mono.wav'
    fs, audio = wav.read(audio_file)
    t = np.arange(len(audio)) * float(1)/fs
    
    plt.figure()
    plt.plot(t,audio)
    plt.grid()

    nfft = 1024
    noverlap=nfft/2
    
    f, t_S, Sxx = signal.spectrogram(audio, fs, window='hamming', nperseg=nfft, 
                                     noverlap=noverlap, nfft=None, detrend='constant', 
                                     return_onesided=True, scaling='density', axis=-1)
    plt.figure()
    plt.pcolormesh(t_S, f, (20*np.log(abs(Sxx)/10)))    
    plt.axis('tight')
