# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import csv

def morph_close(audio, fs=44100, n=1024):
    
    result=np.zeros(np.size(audio))
    l=n/2+1  
    audio_abs=np.abs(audio)
    for i in range(l,len(audio)-l):
        result[i]=np.max(audio_abs[i-l:i+l])
    t = np.arange(len(result)) * (float(n)/fs)  
        
    return result, t, audio_abs
    
def set_threshold(fragment):
    
    silence_file = fragment + '_silence.wav'
    activity_file = fragment + '_activity.wav'
    
    fs, audio_silence = wav.read(silence_file)
    fs, audio_activity = wav.read(activity_file)        
    
    frame_size=512
    
    mc_silence, t_mc_silence, dummy = morph_close(audio_silence, fs, n=frame_size)  
    mc_activity, t_mc_activity, dummy = morph_close(audio_activity, fs, n=frame_size)
    
    plt.figure()
    plt.hist([mc_activity, mc_silence], bins = 200)
    plt.grid()
    plt.axis('tight')
    plt.show()
    
    var_activity = np.var(mc_activity)
    var_silence = np.var(mc_silence)
    
    return var_activity, var_silence
    

if __name__ == "__main__":
    
    dataset=[]    
    cr = csv.reader(open('dataset.csv',"rb"))
    for row in cr:
        dataset.append(row[0]) 

    fragment = dataset[0]    
    
    audio_file = fragment + '_mono.wav'
    gt_file = fragment + '.csv'
    
    fs, audio = wav.read(audio_file)
    t = np.arange(len(audio)) * (1/float(fs))    
    
    audio_closed, t_mc, audio_abs = morph_close(audio, 4*880+1)
    
    plt.figure(figsize=(18,6)) 
    plt.plot(t_mc, audio_abs, 'r', label='wave')  
    plt.plot(t_mc, audio_closed, 'k', label='envelope')
    plt.grid()

    onset=[]
    notes=[]
    cr = csv.reader(open(gt_file,"rb"))
    for row in cr:
        onset.append(row[0]) 
        notes.append(row[1])
    onset = np.array(onset, 'float32')
    i=0
    aux_vad_gt = np.empty([0,], 'int8')
    for note in notes:
        if note=='0':
            aux_vad_gt = np.r_[aux_vad_gt,0]
        else:
            aux_vad_gt = np.r_[aux_vad_gt,1]
        i=i+1
    j=0
    vad_gt = np.empty([len(t),], 'int8')
    for i in range(1,len(onset)):
        while (j<len(t) and t[j]>=onset[i-1] and t[j]<=onset[i]):
            vad_gt[j]=aux_vad_gt[i-1]
            j=j+1     

    plt.figure(figsize=(18,6))
    plt.subplot(2,1,1)
    plt.plot(t,audio)
    plt.plot(t,(2**12)*vad_gt, label='VAD_gt')
    plt.grid()
    plt.title(fragment)
    plt.tight_layout()
    plt.subplot(2,1,2)
    plt.plot(t,audio_closed, label='envelope')
    plt.plot(t,(2**12)*vad_gt, label='VAD_gt')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
#%%
    thershold_activity, thershold_silence = set_threshold(fragment)
