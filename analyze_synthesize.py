# -*- coding: utf-8 -*-
"""
Created on Mon May  9 18:10:49 2016

@author: jbraga
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

if __name__ == "__main__":  

    plt.close("all")
    
    import csv
    dataset=[]    
    cr = csv.reader(open('dataset.csv',"rb"))
    for row in cr:
        dataset.append(row[0]) 
    
    audio_file = dataset[3] + '_mono.wav'
    print audio_file
    fs, audio = wav.read(audio_file)
    t = np.arange(len(audio)) * float(1)/fs
        
#    plt.figure()
#    plt.plot(t,audio)
#    plt.grid()
#    plt.axis('tight')

    nfft = 512
    noverlap=nfft/2
    
    f, t_S, S = signal.spectrogram(audio, fs, window='hamming', nperseg=nfft, 
                                     noverlap=noverlap, nfft=None, detrend='constant',
                                     return_onesided=False, scaling='spectrum', axis=-1, mode='complex')
                                    
    Sxx = np.abs(S) 
    
    plt.figure()
    plt.pcolormesh(t_S, f, 20*np.log(Sxx))
    plt.axis('tight')
                                     
    