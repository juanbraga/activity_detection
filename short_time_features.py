# -*- coding: utf-8 -*-

import numpy as np

def average_energy(audio, fs=44100, n=1024):
    
    ste = np.empty(len(audio)/n)
    for i in range(0,len(audio)/n):
        ste[i] = (np.sum(np.absolute(np.hamming(n)*audio[i*n:(i+1)*n]))/n)
    t = np.arange(len(ste)) * (float(n)/fs)      
    return ste, t
    
def zero_crossing_rate(audio, fs=44100, n=1024):
    
    zcr = np.empty(len(audio)/n)
    audio_sgn = np.sign(audio)    
    for i in range(0,len(audio)/n):
        zcr[i] = (np.sum(np.absolute(np.diff(audio_sgn[i*n:(i+1)*n])))/n)
    t = np.arange(len(zcr)) * (float(n)/fs)    
    return zcr, t