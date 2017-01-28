# -*- coding: utf-8 -*-

import scipy.io.wavfile as wav
import short_time_features as stf
import spectral_features as spf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

#fragment = '../traditional_dataset/density/fragments/density_first_fragment_zoon'

#fragment = '../traditional_dataset/syrinx/fragments/syrinx_first_fragment_douglas'
#fragment = '../traditional_dataset/syrinx/fragments/syrinx_second_fragment_dwyer'
#fragment = '../traditional_dataset/syrinx/fragments/syrinx_third_fragment_rhodes'
fragment = '../traditional_dataset/syrinx/fragments/syrinx_fourth_fragment_bernold'
#fragment = '../traditional_dataset/syrinx/fragments/syrinx_fifth_fragment_bourdin'

#fragment = '../traditional_dataset/allemande/fragments/allemande_first_fragment_nicolet'
#fragment = '../traditional_dataset/allemande/fragments/allemande_second_fragment_gerard'
#fragment = '../traditional_dataset/allemande/fragments/allemande_third_fragment_rampal'
#fragment = '../traditional_dataset/allemande/fragments/allemande_fourth_fragment_larrieu'
#fragment = '../traditional_dataset/allemande/fragments/allemande_fifth_fragment_preston'

audio_file = fragment + '_mono.wav'
gt_file = fragment + '.csv'

fs, audio = wav.read(audio_file)
t = np.arange(len(audio)) * float(1)/fs

frame_size = 512
hop = frame_size
ae, t_ae = stf.average_energy(audio, n=frame_size)
zcr, zcr_frec, t_zcr = stf.zero_crossing_rate(audio, n=frame_size)

import csv
cr = csv.reader(open(gt_file,"rb"))
onset=[]
notes=[]
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

#%%         
plt.figure(figsize=(18,6))
plt.subplot(3,1,1)
plt.plot(t, audio)
plt.grid()
plt.title(fragment)
plt.tight_layout()

plt.subplot(3,1,2)
plt.plot(t_ae, ae/max(ae), 'k', label='stf:average_energy')
plt.plot(t, 0.5*vad_gt, label='VAD_gt')
plt.grid()
plt.xlabel('Time (s)')
plt.legend(loc='best')
plt.tight_layout()

plt.subplot(3,1,3)
plt.plot(t_zcr, zcr/max(zcr), 'k', label='stf:zero_crossing_rate')
plt.plot(t, 0.5*vad_gt, label='VAD_gt')
plt.grid()
plt.xlabel('Time (s)')
plt.legend(loc='best')
plt.tight_layout()

plt.show()

#%%
tness, f0_max, Sxx, f, t_S = spf.tonalness(audio, fs, frame_size, noverlap=(frame_size-hop))

plt.figure(figsize=(18,6))
plt.subplot(3,1,1)
plt.plot(t, audio)
plt.grid()
plt.title(fragment)
plt.tight_layout()

plt.subplot(3,1,2)
plt.plot(t_S, f0_max, label='Spec Max')
#plt.plot(t_zc, f0_gt, label='VAD_gt')
plt.grid()
plt.xlabel('Time (s)')
plt.legend(loc='best')
plt.tight_layout()

plt.subplot(3,1,3)
plt.plot(t_S, tness, label='Tonalness')
plt.plot(t, (max(tness)/2)*vad_gt, label='VAD_gt')
plt.grid()
plt.xlabel('Time (s)')
plt.legend(loc='best')
plt.tight_layout()

plt.show()



