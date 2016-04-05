# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def tonalness(audio, fs=44100, nfft=1024, noverlap=512): 
    
    f, t_S, Sxx = signal.spectrogram(audio, fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)
    
    ind_max = np.argmax(Sxx,axis=0)
    ind_peaks = np.c_[ind_max]
    
    S_harmonics = np.empty(ind_max.shape)
    for i in range(0,len(ind_max)):
        S_harmonics[i]=np.sum(Sxx[ind_peaks[i],i]) 
     
    E = np.sum(Sxx, axis=0) 
    E = E - S_harmonics
    result = np.divide(S_harmonics,E)    
    
    return result, Sxx, f, t_S


if __name__ == "__main__":  

    import csv
    dataset=[]    
    cr = csv.reader(open('dataset.csv',"rb"))
    for row in cr:
        dataset.append(row[0]) 

    total_tness = np.empty([0,])

    for i in range(0,len(dataset)):
    
        audio_file = dataset[i] + '_silence.wav'
        gt_file = dataset[i] + '.csv'
    
        fs, audio = wav.read(audio_file)
        t = np.arange(len(audio)) * (1/44100.0)        
        
        nfft=4096
        overlap=nfft/2
        tness, Sxx, f, t_S = tonalness(audio, fs, nfft, overlap)
        
        total_tness = np.r_[total_tness,tness]
        
    plt.plot(total_tness)
    plt.grid()
    plt.axis('tight')
    
    tness_mean = np.mean(total_tness)
    tness_var = np.var(total_tness)