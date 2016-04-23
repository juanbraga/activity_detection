# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def tonalness(audio, fs=44100, nfft=1024, noverlap=512): 
    
    f, t_S, Sxx = signal.spectrogram(audio, fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)
    
    ind_max = np.argmax(Sxx,axis=0)
    f0_max = f[ind_max]
    ind_peaks = np.c_[ind_max]
    
    S_harmonics = np.empty(ind_max.shape)
    for i in range(0,len(ind_max)):
        S_harmonics[i]=np.sum(Sxx[ind_peaks[i],i]) 
     
    E = np.sum(Sxx, axis=0) 
    E = E - S_harmonics
    result = np.divide(S_harmonics,E)    
    
    return result, f0_max, Sxx, f, t_S


if __name__ == "__main__":  

    import csv
    dataset=[]    
    cr = csv.reader(open('dataset.csv',"rb"))
    for row in cr:
        dataset.append(row[0]) 

    total_tness_activity = np.empty([0,])
    total_tness_silence = np.empty([0,])

    for i in range(0,len(dataset)):
    
        silence_file = dataset[i] + '_silence.wav'
        activity_file = dataset[i] + '_activity.wav'
    
        fs, audio_silence = wav.read(silence_file)
        fs, audio_activity = wav.read(activity_file)        
        
        nfft=4096
        overlap=nfft/2
        
        tness_activity, dummy, Sxx, f, t_S = tonalness(audio_activity, fs, nfft, overlap)
        tness_silence, dummy, Sxx, f, t_S = tonalness(audio_silence, fs, nfft, overlap)
        
        total_tness_activity = np.r_[total_tness_activity,tness_activity]
        total_tness_silence = np.r_[total_tness_silence,tness_silence]
        
    plt.figure()        
    plt.hist([total_tness_activity, total_tness_silence], bins = 200)
    plt.axis('tight')
    