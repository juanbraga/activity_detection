# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def tonalness(audio, fs=44100, nfft=1024, noverlap=512):
    
    f, t_S, Sxx = signal.spectrogram(audio, fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)
    
    ind_max = np.argmax(Sxx,axis=0)
    ind_peaks = np.c_[ind_max, 2*ind_max, 3*ind_max, 4*ind_max]
    
    S_harmonics = np.empty(ind_max.shape)
    for i in range(0,len(ind_max)):
        S_harmonics[i]=np.sum(Sxx[ind_peaks[i],i]) 
     
    E = np.sum(Sxx, axis=0) 
    E = E - S_harmonics
    result = np.divide(S_harmonics,E)    
    
    
    return result, Sxx, f, t_S

if __name__ == "__main__":  

    #fragment = '../traditional_dataset/density/fragments/density_first_fragment_zoon'
    
    #fragment = '../traditional_dataset/syrinx/fragments/syrinx_first_fragment_douglas'
    #fragment = '../traditional_dataset/syrinx/fragments/syrinx_second_fragment_dwyer'
    #fragment = '../traditional_dataset/syrinx/fragments/syrinx_third_fragment_rhodes'
    #fragment = '../traditional_dataset/syrinx/fragments/syrinx_fourth_fragment_bernold'
    #fragment = '../traditional_dataset/syrinx/fragments/syrinx_fifth_fragment_bourdin'
    
    #fragment = '../traditional_dataset/allemande/fragments/allemande_first_fragment_nicolet'
    #fragment = '../traditional_dataset/allemande/fragments/allemande_second_fragment_gerard'
    #fragment = '../traditional_dataset/allemande/fragments/allemande_third_fragment_rampal'
    #fragment = '../traditional_dataset/allemande/fragments/allemande_fourth_fragment_larrieu'
    fragment = '../traditional_dataset/allemande/fragments/allemande_fifth_fragment_preston'
    
    audio_file = fragment + '_mono.wav'
    gt_file = fragment + '.csv'
    
    fs, audio = wav.read(audio_file)
    t = np.arange(len(audio)) * (1/44100.0)        
    
    nfft=4096
    overlap=nfft/2
    tonalness, Sxx, f, t_S = tonalness(audio, fs, nfft, overlap)
    
    import csv
    cr = csv.reader(open(gt_file,"rb"))
    onset=[]
    notes=[]
    for row in cr:
        onset.append(row[0]) 
        notes.append(row[1])
    onset = np.array(onset, 'float32')
    i=0
    aux_vad_gt = np.empty([0,])
    for note in notes:
        if note=='0':
            aux_vad_gt = np.r_[aux_vad_gt,0]
        else:
            aux_vad_gt = np.r_[aux_vad_gt,1]
        i=i+1
    j=0
    vad_gt = np.empty([len(t),], 'float32')
    for i in range(1,len(onset)):
        while (j<len(t) and t[j]>=onset[i-1] and t[j]<=onset[i]):
            vad_gt[j]=aux_vad_gt[i-1]
            j=j+1    
    
    plt.figure(figsize=(18,6))
    plt.subplot(2,1,1)    
    plt.pcolormesh(t_S, f, np.log(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.axis('tight')
    plt.subplot(2,1,2)
    plt.plot(t_S,tonalness)
    plt.plot(t, vad_gt, label='VAD_gt')
    plt.grid()
    plt.axis('tight')
    plt.show()