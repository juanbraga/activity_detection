# -*- coding: utf-8 -*-

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


plt.close('all')

fragment = '../traditional_dataset/density/fragments/density_first_fragment_zoon'

#fragment = '../traditional_dataset/syrinx/fragments/syrinx_first_fragment_douglas'
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

audio, sr = librosa.load(audio_file, sr=44100, mono=True)
t = np.arange(len(audio)) * (1/44100.0)

frame_size = 8192
hop = frame_size / 2
rms = librosa.feature.rmse(audio, n_fft=frame_size, hop_length=hop)
rms = rms.T
t_rms = np.arange(len(rms)) * (hop/44100.0)

frame_size = 2756
hop = 1024
zero_crossing = librosa.feature.zero_crossing_rate(audio, frame_length = frame_size, hop_length = hop)
zero_crossing = zero_crossing.T
t_zc = np.arange(len(zero_crossing)) * (hop/44100.0)

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
f0_gt = np.empty([len(t_zc),],'float64')
for i in range(1,len(onset)):
    while (j<len(t_zc) and t_zc[j]>=onset[i-1] and t_zc[j]<=onset[i]):
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
plt.plot(t_zc, f0_gt, 'r', label='f0_gt')
plt.plot(t_zc, 44100*zero_crossing/2, 'k', label='zero_crossing')
plt.grid()
plt.xlabel('Time (s)')
plt.legend(loc='best')
plt.tight_layout()

plt.show()

#%%

#alpha = 1
#b = np.array([1, -alpha])
#w, h = signal.freqz(b)
#
#import matplotlib.pyplot as plt
#fig = plt.figure()
#plt.title('Digital filter frequency response')
#ax1 = fig.add_subplot(111)
#
#plt.plot(w*(22050/(2*np.pi)), 20 * np.log10(abs(h)), 'b')
#plt.ylabel('Amplitude [dB]', color='b')
#plt.xlabel('Frequency [rad/sample]')
#
#ax2 = ax1.twinx()
#angles = np.unwrap(np.angle(h))
#plt.plot(w, angles, 'g')
#plt.ylabel('Angle (radians)', color='g')
#plt.grid()
#plt.axis('tight')
#plt.show()

#%%

f, t_S, Sxx = signal.spectrogram(audio, sr, window='hamming', nperseg=1024, noverlap=512, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)

S_max = np.amax(Sxx,axis=0)
ind_max = np.argmax(Sxx,axis=0)
f0_max = f[ind_max]

#plt.figure()
#plt.pcolormesh(t, f, np.log(Sxx))
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.plot(t, f0_max)
#plt.axis('tight')
#plt.show()

#%%

E = np.sum(Sxx, axis=0)
E = E - S_max
tonalness = np.divide(S_max,E)

plt.figure(figsize=(18,6))
plt.subplot(3,1,1)
plt.plot(t, audio)
plt.grid()
plt.title(fragment)
plt.tight_layout()

plt.subplot(3,1,2)
plt.plot(t_S, f0_max, label='Spec Max')
plt.plot(t_zc, f0_gt, label='VAD_gt')
plt.grid()
plt.xlabel('Time (s)')
plt.legend(loc='best')
plt.tight_layout()

plt.subplot(3,1,3)
plt.plot(t_S, tonalness, label='Harmonicity')
plt.plot(t_rms, (max(tonalness)/2)*vad_gt, label='VAD_gt')
plt.grid()
plt.xlabel('Time (s)')
plt.legend(loc='best')
plt.tight_layout()

plt.show()

#%% 

ar = librosa.autocorrelate(audio, max_size = 1024)

#%%
plt.figure()
plt.plot(ar)



