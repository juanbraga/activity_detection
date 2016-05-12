# -*- coding: utf-8 -*-
"""
Created on Mon May  9 18:10:49 2016

@author: jbraga
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.fftpack as fft

def stft(x, fs, win, hop):
    pass

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
    noverlap=(3*nfft)/4
    
    f, t_S, S = signal.spectrogram(audio, fs, window='hamming', nperseg=nfft, 
                                     noverlap=noverlap, nfft=None, detrend='constant',
                                     return_onesided=True, scaling='density', axis=-1, mode='complex')
                                    
    Sxx = np.abs(S) 
    
    plt.figure()
    plt.pcolormesh(t_S, f, 20*np.log(Sxx))
    plt.axis('tight')
    
    X = fft.fft(audio)
    X_r = fft.rfft(audio)
    
    #%%
    plt.figure(figsize=(18,6))
    plt.plot(abs(X_r),label='rfft')
    plt.plot(abs(X),label='fft')
    plt.legend(loc='best')
    plt.grid()
    plt.axis('tight')
                                     
    