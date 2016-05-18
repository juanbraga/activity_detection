# -*- coding: utf-8 -*-
"""
Created on Mon May  9 18:10:49 2016

@author: jbraga
"""

import numpy as np
import stochastic_spectrum_estimation as sse
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def stft(x, fs, win, hop):
    
    S = np.empty([(win),(len(x)/hop)+1],dtype='complex64')
    
    for i in range(0,(len(x)/hop)+1):    
        frame=x[i*hop:i*hop+win]
        frame = np.hamming(len(frame))*frame        
        X = np.fft.fft(frame,nfft)
        S[:,i] = X 
        
    freq = np.fft.fftfreq(win, 1/float(fs))
    t = np.arange((len(x)/hop)+1) * float(hop)/fs   
    
    return S, freq, t
    

def istft(X,fs,win,hop,filename="resynth.wav"):
    
    resynth = np.empty([hop*(X.shape[1]-1)+win,], dtype='float64')
    for i in range(0,X.shape[1]):    
        iX = np.fft.ifft(X[:,i])
        aux_resynth = np.zeros(hop*(X.shape[1]-1)+win, dtype='float64')
        aux_resynth[i*hop:i*hop+win]=iX.real
        resynth = resynth + aux_resynth 
    
    resynth_int16 = np.array(resynth,dtype='int16')
    wav.write(filename, fs, resynth_int16)
    t=np.arange(hop*X.shape[1]) * float(1)/fs
    
    return resynth, t 
    
def P2R(radii, angles):
    return radii * np.exp(1j*angles)

def R2P(x):
    return np.abs(x), np.angle(x)

if __name__ == "__main__":  

    plt.close("all")
    
    import csv
    dataset=[]    
    cr = csv.reader(open('dataset.csv',"rb"))
    for row in cr:
        dataset.append(row[0]) 
    
    audio_file = dataset[4] + '_mono.wav'
    print audio_file
    fs, audio = wav.read(audio_file)
    t = np.arange(len(audio)) * float(1)/fs
        
#    plt.figure()
#    plt.plot(t,audio)
#    plt.grid()
#    plt.axis('tight')

    nfft = 512
    noverlap=(3*nfft)/4
    hop = nfft-noverlap
    
#    f, t_S, S = signal.spectrogram(audio, fs, window='hamming', nperseg=nfft, 
#                                     noverlap=noverlap, nfft=None, detrend='constant',
#                                     return_onesided=False, scaling='density', axis=-1, mode='complex')
#                                    
#    Sxx = np.abs(S) 
#    
#    plt.figure()
#    plt.subplot(2,1,1)  
#    plt.pcolormesh(t_S, f, 20*np.log(np.abs(S)))
#    plt.axis('tight')
#    plt.subplot(2,1,2)
#    plt.pcolormesh(t_S, f, np.angle(S))
#    plt.axis('tight')            
#
#    spec, f_spec, t_spec = stft(audio, fs, nfft, (nfft-noverlap))
#    resynth, t_resynth = istft(spec,fs,nfft,(nfft-noverlap))
#    
#    plt.figure()
#    plt.subplot(2,1,1)
#    plt.pcolormesh(t_spec, f_spec, 20*np.log(np.abs(spec)))
#    plt.axis('tight')    
#    plt.subplot(2,1,2) 
#    plt.pcolormesh(t_spec, f_spec, np.angle(spec))
#    plt.axis('tight')
     
    S, f, t_S = stft(audio, fs, nfft, (nfft-noverlap))
    SSE, env = sse.sse(np.abs(S))
    
#%%
    plt.figure()
    plt.subplot(2,1,1)
    plt.pcolormesh(t_S, f, 20*np.log(np.abs(S)))
    plt.axis('tight')    
    plt.subplot(2,1,2) 
    plt.pcolormesh(t_S, f, 20*np.log(SSE))
    plt.axis('tight')
    
#%%    
    S_resynth = P2R(SSE,np.angle(S))
    resynth, t_resynth = istft(S_resynth,fs,nfft,(nfft-noverlap))