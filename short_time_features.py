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

#    fragment = '../traditional_dataset/density/fragments/density_first_fragment_zoon'
    
#    fragment = '../traditional_dataset/syrinx/fragments/syrinx_first_fragment_douglas'
#    fragment = '../traditional_dataset/syrinx/fragments/syrinx_second_fragment_dwyer'
    fragment = '../traditional_dataset/syrinx/fragments/syrinx_third_fragment_rhodes'
#    fragment = '../traditional_dataset/syrinx/fragments/syrinx_fourth_fragment_bernold'
#    fragment = '../traditional_dataset/syrinx/fragments/syrinx_fifth_fragment_bourdin'
    
#    fragment = '../traditional_dataset/allemande/fragments/allemande_first_fragment_nicolet'
#    fragment = '../traditional_dataset/allemande/fragments/allemande_second_fragment_gerard'
#    fragment = '../traditional_dataset/allemande/fragments/allemande_third_fragment_rampal'
#    fragment = '../traditional_dataset/allemande/fragments/allemande_fourth_fragment_larrieu'
#    fragment = '../traditional_dataset/allemande/fragments/allemande_fifth_fragment_preston'
    
    audio_file = fragment + '_mono.wav'
    gt_file = fragment + '.csv'
    
    fs, audio = wav.read(audio_file)
    t = np.arange(len(audio)) * (1/44100.0)        
    
    frame_size=1024
    
    zcr, t_zcr = zero_crossing_rate(audio, fs, n=frame_size)   
    ae, t_ae = average_energy(audio, fs, n=frame_size)
        
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
    vad_gt = np.empty([len(t),], 'int8')
    for i in range(1,len(onset)):
        while (j<len(t) and t[j]>=onset[i-1] and t[j]<=onset[i]):
            vad_gt[j]=aux_vad_gt[i-1]
            j=j+1    
    
    plt.figure()
    plt.plot(t_zcr, zcr/float(max(zcr)), label='zero crossing rate')
    plt.plot(t_ae, ae/float(max(ae)),label='average energy')    
    plt.plot(t, vad_gt, label='activity detection (gt)')    
    plt.grid()    
    plt.ylabel('Amplitude')
    plt.xlabel('Time [sec]')
    plt.axis('tight')
    plt.legend(loc='best')
    plt.show()
    
    #%%
    
    plt.figure()
    plt.plot(t_zcr[0:(len(t_zcr)-1)], np.diff(zcr), label='zero crossing rate')   
    plt.plot(t, vad_gt, label='activity detection (gt)')    
    plt.grid()    
    plt.ylabel('Amplitude')
    plt.xlabel('Time [sec]')
    plt.axis('tight')
    plt.legend(loc='best')
    plt.show()