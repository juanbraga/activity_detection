# -*- coding: utf-8 -*-

import librosa
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

#fragment = '../traditional_dataset/syrinx/fragments/syrinx_first_fragment_douglas'
#fragment = '../traditional_dataset/syrinx/fragments/syrinx_second_fragment_dwyer'
fragment = '../traditional_dataset/syrinx/fragments/syrinx_third_fragment_rhodes'
#fragment = '../traditional_dataset/syrinx/fragments/syrinx_fourth_fragment_bernold'
#fragment = '../traditional_dataset/syrinx/fragments/syrinx_fifth_fragment_bourdin'

#fragment = '../traditional_dataset/allemande/fragments/allemande_first_fragment_nicolet'
#fragment = '../traditional_dataset/allemande/fragments/allemande_second_fragment_gerard'
#fragment = '../traditional_dataset/allemande/fragments/allemande_third_fragment_rampal'
#fragment = '../traditional_dataset/allemande/fragments/allemande_fourth_fragment_larrieu'
#fragment = '../traditional_dataset/allemande/fragments/allemande_fifth_fragment_preston'

audio_file = fragment + '_mono.wav'
gt_file = fragment + '.csv'

audio, sr = librosa.load(audio_file, sr=44100, mono=True)

t = np.arange(len(audio)) * (1/44100.0)

frame_size = 8192
hop = frame_size / 2
rms = librosa.feature.rmse(audio, n_fft=frame_size, hop_length=hop)
rms = rms.T
t_rms = np.arange(len(rms)) * (hop/44100.0)

#%%

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
vad_gt = np.empty([len(t_rms),], 'float32')
for i in range(1,len(onset)):
    while (j<len(t_rms) and t_rms[j]>=onset[i-1] and t_rms[j]<=onset[i]):
        vad_gt[j]=aux_vad_gt[i-1]
        j=j+1      
        

cr = csv.reader(open("../pitch_extraction/note_convertion.csv","rb"))
notation=[]
frequency=[]
for row in cr:
    notation.append(row[0]) 
    frequency.append(row[1])
frequency = np.array(frequency, 'float64')
i=0
aux_f0_gt = np.empty([0,])
for note in notes:
    if note=='0':
        aux_f0_gt = np.r_[aux_f0_gt,0]
    else:
        aux_f0_gt = np.r_[aux_f0_gt,frequency[notation.index(note)]]
    i=i+1
j=0
f0_gt = np.empty([len(t_rms),],'float64')
for i in range(1,len(onset)):
    while (j<len(t_rms) and t_rms[j]>=onset[i-1] and t_rms[j]<=onset[i]):
        f0_gt[j]=aux_f0_gt[i-1]
        j=j+1 

plt.figure(figsize=(18,6))
plt.subplot(3,1,1)
plt.plot(t, audio)
plt.grid()
plt.title(fragment)
plt.tight_layout()

plt.subplot(3,1,2)
plt.plot(t_rms, rms, label='RMS Energy')
plt.plot(t_rms, (max(rms)/2)*vad_gt, label='VAD_gt')
plt.grid()
plt.xlabel('Time (s)')
plt.legend(loc='best')
plt.tight_layout()

plt.subplot(3,1,3)
plt.plot(t_rms, f0_gt, 'r', label='f0_gt')
plt.grid()
plt.xlabel('Time (s)')
plt.legend(loc='best')
plt.tight_layout()

plt.show()
