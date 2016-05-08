# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 21:05:21 2016

@author: jbraga
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
<<<<<<< .merge_file_a03092
import scipy.io 
=======
import short_time_features as stf
>>>>>>> .merge_file_a07504

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
    env = np.sqrt(Eg/M.shape[0])
    
    return env
    
           
if __name__ == "__main__":  

    plt.close("all")
    
    import csv
    dataset=[]    
    cr = csv.reader(open('dataset.csv',"rb"))
    for row in cr:
        dataset.append(row[0]) 
    
    audio_file = dataset[6] + '_mono.wav'
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
<<<<<<< .merge_file_a03092
                                     return_onesided=True, scaling='density', axis=-1)
    
=======
                                     return_onesided=True, scaling='spectrum', axis=-1)
>>>>>>> .merge_file_a07504
#    plt.figure()
#    plt.pcolormesh(t_S, f, 20*np.log(Sxx))
#    plt.axis('tight')
#    plt.show()
    
<<<<<<< .merge_file_a03092
    Sxx_filt = mean_filter(Sxx, n=3)
    
#    plt.figure()
#    plt.pcolormesh(t_S, f, 20*np.log(Sxx_filt))
#    plt.axis('tight')
#    plt.show()
    
#%%    
    Sxx_filt_inv = reciprocal(Sxx_filt)
    
#    plt.figure()
#    plt.pcolormesh(t_S, f, 20*np.log(Sxx_filt_inv))
#    plt.axis('tight')
#    plt.show()
    
#%%
   
    Sxx_filt_inv_filt = mean_filter(Sxx_filt_inv, n=25)
    
#    plt.figure()
#    plt.pcolormesh(t_S, f, 20*np.log(Sxx_filt_inv_filt))
#    plt.axis('tight')
#    plt.show()
    
#%%
   
    SSE = reciprocal(Sxx_filt_inv_filt)
    
#    plt.figure()
#    plt.pcolormesh(t_S, f, 20*np.log(SSE))
#    plt.axis('tight')
#    plt.show()
    
##%%
#    plt.figure(figsize=(18,6))
#    plt.plot(f, Sxx[:,1000], color='red')
#    plt.plot(f, SSE[:,1000], color='blue')

#%%
    env = time_env(SSE)    
#    plt.figure(figsize=(18,6))  
#    plt.plot(t,audio)    
#    plt.plot(t_S, env*max(audio), color='red')
#    plt.grid()

#%%
#    plt.figure()
#    plt.subplot(4,1,(1,3))
#    plt.pcolormesh(t_S[(2<t_S) & (t_S<8)], f, 20*np.log(Sxx[:,(2<t_S) & (t_S<8)]))
#    plt.axis('tight')
#    plt.subplot(4,1,4)
#    plt.plot(t_S[(2<t_S) & (t_S<8)], env[(2<t_S) & (t_S<8)], color='red')
#    plt.axis('tight')
#    plt.grid()
#    plt.show()    


#%%
#    plt.figure()
#    plt.subplot(4,1,(1,3))
#    plt.pcolormesh(t_S, f, 20*np.log(Sxx))
#    plt.axis('tight')
#    plt.subplot(4,1,4)
#    plt.plot(t_S, env, color='red')
#    plt.axis('tight')
#    plt.grid()
#    plt.show()

#%%    

#    plt.figure(figsize=(18,6))  
#    plt.plot(t_S, env, color='red')
#    plt.grid()
    

#%%    
    env_filt = mean_filter(env, n=4)
#    plt.figure(figsize=(18,6))  
#    plt.plot(t_S, env_filt, color='red')
#    plt.grid()
    
#%%
#    plt.figure()
#    plt.subplot(4,1,1)
#    plt.pcolormesh(t_S, f[f<10000], 20*np.log(Sxx[f<10000,:]))
#    plt.axis('tight')
#    plt.subplot(4,1,(2,4))
#    plt.plot(t_S, env_filt, color='red')
#    plt.axis('tight')
#    plt.grid()
#    plt.show()
    
    #%%
    np.savetxt("sse_env.csv", np.c_[t_S, env], delimiter=",")

    #%%

#    plt.figure()
#    plt.pcolormesh(t_S, f, 20*np.log(Sxx-SSE))
#    plt.axis('tight')
#    plt.show()    
    
    #%%
    scipy.io.savemat('sse', mdict={'sse':SSE,'t':t_S,'f':f,'win':np.hamming(nfft),'noverlap':noverlap})
=======
    SSE, env = sse(Sxx)    
    Ew = np.sum(np.hamming(Sxx.shape[0])**2)
    env_spec = time_env(Sxx)
    env_spec2 = np.sqrt(np.sum(Sxx,axis=0)/(Sxx.shape[0]*Ew))
    env_spec3, t = stf.average_energy(audio,fs,nfft)
    
    plt.figure()
    plt.plot(t_S,env_spec,color='green')
    plt.plot(t_S,env_spec2,color='red')
    plt.plot(t_S,env_spec3,color='black')

    plt.figure()
    plt.subplot(4,1,(1,3))
    plt.pcolormesh(t_S, f, 20*np.log(SSE))
    plt.axis('tight')
    plt.subplot(4,1,4)
    plt.plot(t_S, env, color='red')
    plt.plot(t_S, env_spec, color='green')
    plt.axis('tight')
    plt.grid()
    plt.show()

   
    env_filt = mean_filter(env, n=4)
    np.savetxt("sse_env.csv", np.c_[t_S, env], delimiter=",")

>>>>>>> .merge_file_a07504
