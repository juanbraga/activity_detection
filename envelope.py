# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav

def morph_close(audio, length):
    
    result=np.zeros(np.size(audio))
    l=length/2+1  
    audio_abs=np.abs(audio)
    for i in range(l,len(audio)-l):
        result[i]=np.max(audio_abs[i-l:i+l])
        
    return result, audio_abs


if __name__ == "__main__":
    
    fragment = '../traditional_dataset/density/fragments/density_first_fragment_zoon'
    
#    fragment = '../traditional_dataset/syrinx/fragments/syrinx_first_fragment_douglas'
    #fragment = '../traditional_dataset/syrinx/fragments/syrinx_second_fragment_dwyer'
    #fragment = '../traditional_dataset/syrinx/fragments/syrinx_third_fragment_rhodes'
    #fragment = '../traditional_dataset/syrinx/fragments/syrinx_fourth_fragment_bernold'
    #fragment = '../traditional_dataset/syrinx/fragments/syrinx_fifth_fragment_bourdin'
    
    #fragment = '../traditional_dataset/allemande/fragments/allemande_first_fragment_nicolet'
    #fragment = '../traditional_dataset/allemande/fragments/allemande_second_fragment_gerard'
    #fragment = '../traditional_dataset/allemande/fragments/allemande_third_fragment_rampal'
    #fragment = '../traditional_dataset/allemande/fragments/allemande_fourth_fragment_larrieu'
    #fragment = '../traditional_dataset/allemande/fragments/allemande_fifth_fragment_preston'
    
    audio_file = fragment + '_mono.wav'
    gt_file = fragment + '.csv'
    
    fs, audio = wav.read(audio_file)
    t = np.arange(len(audio)) * (1/44100.0)    
    
    audio_closed, audio_abs = morph_close(audio, 4*880+1)
    
    
    plt.figure(figsize=(18,6)) 
    plt.plot(t, audio_abs, 'r', label='wave')  
    plt.plot(t, audio_closed, 'k', label='envelope')
    plt.grid()
    
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
            
#%%    
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
