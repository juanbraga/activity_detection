# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 21:05:21 2016

@author: jbraga
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.io 
import short_time_features as stf

def mean_filter(M, n=20):
    
    filtered = np.empty(M.shape)
    mean_filt = (1/float(n))*np.ones((n,))
    for i in range(0,M.shape[1]):
        filtered[:,i] = np.convolve(M[:,i],mean_filt,'same')
        
    return filtered

def reciprocal(M):
    inv = 1/M
    return inv
    
def sse(M):
    
    aux = mean_filter(M, n=3)
    aux = reciprocal(aux)
    aux = mean_filter(aux, n=25)
    SSE = reciprocal(aux)

    env = time_env(SSE)  
    
    return SSE, env
  
    
def time_env(M):
        
    Ew = np.sum(np.hamming(M.shape[0])**2)
    EB = np.empty((M.shape[1],1))    
    for i in range(0,M.shape[1]):   
        EB[i] = np.sum(M[:,i])
    Eg = EB/Ew
    aux_env = np.sqrt(Eg/M.shape[0])
    
#    aux_env2 = np.sqrt(np.sum(M,axis=0)/(M.shape[0]*Ew))  
    
    return aux_env #, aux_env2 
    
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
    noverlap=0#3*(nfft/4)
    
    f, t_S, Sxx = signal.spectrogram(audio, fs, window='hamming', nperseg=nfft, 
                                     noverlap=noverlap, nfft=None, detrend='constant',
                                     return_onesided=True, scaling='spectrum', axis=-1)
#    plt.figure()
#    plt.pcolormesh(t_S, f, 20*np.log(Sxx))
#    plt.axis('tight')
#    plt.show()
    
    SSE, sse_env = sse(Sxx)    
    sig_env = time_env(Sxx)
    sig_env2, t_dummy = stf.average_energy(audio,fs,nfft)

    plt.figure(figsize=(18,6))
    plt.plot(t_S,20*np.log(sig_env/max(sig_env)),color='green',label='spectral energy')
    plt.plot(t_S,20*np.log(sig_env2/max(sig_env2)),color='black',label='rms')
    plt.grid()
    plt.legend(loc='best')
    plt.axis('tight')

#%%

    plt.figure()
    plt.subplot(5,1,(1,2))
    plt.pcolormesh(t_S, f, 20*np.log(Sxx))
    plt.axis('tight')

    plt.subplot(5,1,(3,4))
    plt.pcolormesh(t_S, f, 20*np.log(SSE))
    plt.axis('tight')    
    plt.subplot(5,1,5)
    plt.plot(t_S, (sse_env), color='red', label='sse_env')
#    plt.plot(t_S, (sig_env), color='green', label='sig_env')
    plt.legend(loc='upper left')
    plt.axis('tight')
    plt.grid()
    plt.show()

    np.savetxt("sse_env.csv", np.c_[t_S, sse_env, sig_env], delimiter=",")
    scipy.io.savemat('sse', mdict={'sse':SSE,'t':t_S,'f':f,'win':np.hamming(nfft),'noverlap':noverlap})
