# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 21:05:21 2016

@author: jbraga
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def moving_average(a, n=3) :
    
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    
    return ret[n - 1:] / n

def short_filter(M, n=3):

#    filtered = np.zeros(M.shape, 'float32')    
    filtered = M.copy() 
    
    for i in range(0,M.shape[1]):
        for j in range(n/2,M.shape[0]-n/2):
            filtered[j,i]=np.sum(M[j-1:j+2,i])/float(n)
            
    return filtered
    
def reciprocal(M):
    inv = 1/M
    return inv
           

if __name__ == "__main__":  

    import csv
    dataset=[]    
    cr = csv.reader(open('dataset.csv',"rb"))
    for row in cr:
        dataset.append(row[0]) 
    
    audio_file = dataset[0] + '_mono.wav'
    fs, audio = wav.read(audio_file)
    t = np.arange(len(audio)) * float(1)/fs
    plt.close("all")
    
    plt.figure()
    plt.plot(t,audio)
    plt.grid()
    plt.axis('tight')

    nfft = 1024
    noverlap=nfft/2
    
    f, t_S, Sxx = signal.spectrogram(audio, fs, window='hamming', nperseg=nfft, 
                                     noverlap=noverlap, nfft=None, detrend='constant', 
                                     return_onesided=True, scaling='density', axis=-1)
    plt.figure()
    plt.pcolormesh(t_S, f, 20*np.log(Sxx))
    plt.axis('tight')
    plt.show()
    
    Sxx_filt = short_filter(Sxx.copy())
    
    plt.figure()
    plt.pcolormesh(t_S, f, 20*np.log(Sxx_filt))
    plt.axis('tight')
    plt.show()
    
#%%    
    Sxx_filt_inv = reciprocal(Sxx_filt)
    
    plt.figure()
    plt.pcolormesh(t_S, f, 20*np.log(Sxx_filt_inv))
    plt.axis('tight')
    plt.show()