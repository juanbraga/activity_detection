# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def zero_crossing_rate(audio, fs=44100, n=1024):
    
    result = np.empty(len(audio)/n)
    audio_sgn = np.sign(audio)    
    for i in range(0,len(audio)/n):
        result[i] = (np.sum(np.absolute(np.diff(audio_sgn[i*n:(i+1)*n])))/float(n))
    t = np.arange(len(result)) * (float(n)/fs)    
    return result, t

def average_energy(audio, fs=44100, n=1024):
    
    result = np.empty(len(audio)/n)
    for i in range(0,len(audio)/n):
        result[i] = (np.sum(np.absolute(np.hamming(n)*audio[i*n:(i+1)*n]))/float(n))
    t = np.arange(len(result)) * (float(n)/fs)      
    return result, t
   
def morph_close(audio, fs=44100, n=1024):
    
    result=np.zeros(np.size(audio))
    l=n/2+1  
    audio_abs=np.abs(audio)
    for i in range(l,len(audio)-l):
        result[i]=np.max(audio_abs[i-l:i+l])
    t = np.arange(len(result)) * (float(n)/fs)  
        
    return result, t, audio_abs

    
if __name__ == "__main__":  

    import csv
    dataset=[]    
    cr = csv.reader(open('dataset.csv',"rb"))
    for row in cr:
        dataset.append(row[0]) 
    
    total_zcr_silence = np.empty([0,])
    total_ae_silence = np.empty([0,])
    total_mc_silence = np.empty([0,])
    total_zcr_activity = np.empty([0,])
    total_ae_activity = np.empty([0,])
    total_mc_activity = np.empty([0,])

    for i in range(0,len(dataset)):
    
        silence_file = dataset[i] + '_silence.wav'
        activity_file = dataset[i] + '_activity.wav'
        
        fs, audio_silence = wav.read(silence_file)
        fs, audio_activity = wav.read(activity_file)        
        
        frame_size=1024
        
        zcr_silence, t_zcr_silence = zero_crossing_rate(audio_silence, fs, n=frame_size)  
        zcr_activity, t_zcr_activity = zero_crossing_rate(audio_activity, fs, n=frame_size) 
        ae_silence, t_ae_silence = average_energy(audio_silence, fs, n=frame_size)  
        ae_activity, t_ae_activity = average_energy(audio_activity, fs, n=frame_size) 
        mc_silence, t_mc_silence, dummy = morph_close(audio_silence, fs, n=frame_size)  
        mc_activity, t_mc_activity, dummy = morph_close(audio_activity, fs, n=frame_size) 
        
        total_zcr_silence = np.r_[total_zcr_silence,zcr_silence]
        total_ae_silence = np.r_[total_ae_silence,ae_silence]
        total_mc_silence = np.r_[total_mc_silence,mc_silence]
        total_zcr_activity = np.r_[total_zcr_activity,zcr_activity]
        total_ae_activity = np.r_[total_ae_activity,ae_activity]
        total_mc_activity = np.r_[total_mc_activity,mc_activity]


#    plt.figure()    
#    plt.subplot(3,1,1)    
#    plt.plot(total_ae_activity)
#    plt.plot(total_ae_silence)
#    plt.grid()
#    plt.axis('tight')
#    plt.subplot(3,1,2)    
#    plt.plot(total_zcr_activity)
#    plt.plot(total_zcr_silence)    
#    plt.grid()
#    plt.axis('tight')    
#    plt.subplot(3,1,3)    
#    plt.plot(total_mc_activity)
#    plt.plot(total_mc_silence)
#    plt.grid()
#    plt.axis('tight')    

#%%
    plt.figure()    
    plt.subplot(3,1,1)    
    plt.hist([total_zcr_activity, total_zcr_silence], bins = 200)
    plt.axis('tight')
    plt.subplot(3,1,2)    
    plt.hist([total_ae_activity, total_ae_silence], bins = 200)
    plt.axis('tight')    
    plt.subplot(3,1,3)    
    plt.hist([total_mc_activity, total_mc_silence], bins = 200)
