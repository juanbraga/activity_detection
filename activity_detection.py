
# -*- coding: utf-8 -*-

import librosa
import short_time_features as stf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

plt.close('all')

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

audio, sr = librosa.load(audio_file, sr=44100, mono=True)
t = np.arange(len(audio)) * (1/44100.0)

#%%

frame_size = 512
hop = frame_size
rms = librosa.feature.rmse(audio, n_fft=frame_size, hop_length=hop)
rms = rms.T
t_rms = np.arange(len(rms)) * (hop/44100.0)

ae, t_ae = stf.average_energy(audio, n=frame_size)

frame_size = 512
hop = frame_size
zero_crossing = librosa.feature.zero_crossing_rate(audio, frame_length = frame_size, hop_length = hop)
zero_crossing = zero_crossing.T
t_zero_crossing = np.arange(len(zero_crossing)) * (hop/44100.0)

zcr, t_zcr = stf.zero_crossing_rate(audio, n=frame_size)

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
         

plt.figure(figsize=(18,6))
plt.subplot(3,1,1)
plt.plot(t, audio)
plt.grid()
plt.title(fragment)
plt.tight_layout()

plt.subplot(3,1,2)
plt.plot(t_rms, rms/max(rms), 'r', label='librosa:rmse')
plt.plot(t_ae, ae/max(ae), 'k', label='stf:average_energy')
plt.plot(t_rms, 0.5*vad_gt, label='VAD_gt')

plt.grid()
plt.xlabel('Time (s)')
plt.legend(loc='best')
plt.tight_layout()

plt.subplot(3,1,3)
plt.plot(t_zero_crossing, zero_crossing/max(zero_crossing), 'r', label='librosa:zero_crossing_rate')
plt.plot(t_zcr, zcr/max(zcr), 'k', label='stf:zero_crossing_rate')
plt.plot(t_rms, 0.5*vad_gt, label='VAD_gt')
plt.grid()
plt.xlabel('Time (s)')
plt.legend(loc='best')
plt.tight_layout()

plt.show()

#%%

plt.figure(figsize=(18,6))
plt.subplot(2,1,1)
plt.plot(t, audio)
plt.grid()
plt.title(fragment)
plt.tight_layout()

plt.subplot(2,1,2)
plt.plot(t_zcr, zcr/max(zcr), 'r', label='stf:zcr')
plt.plot(t_ae, ae/max(ae), 'k', label='stf:ae')
plt.plot(t_rms, 0.5*vad_gt, label='gt:vad')
plt.grid()
plt.xlabel('Time (s)')
plt.legend(loc='best')
plt.tight_layout()


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

#E = np.sum(Sxx, axis=0)
#E = E - S_max
#tonalness = np.divide(S_max,E)
#
#plt.figure(figsize=(18,6))
#plt.subplot(3,1,1)
#plt.plot(t, audio)
#plt.grid()
#plt.title(fragment)
#plt.tight_layout()
#
#plt.subplot(3,1,2)
#plt.plot(t_S, f0_max, label='Spec Max')
#plt.plot(t_zc, f0_gt, label='VAD_gt')
#plt.grid()
#plt.xlabel('Time (s)')
#plt.legend(loc='best')
#plt.tight_layout()
#
#plt.subplot(3,1,3)
#plt.plot(t_S, tonalness, label='Harmonicity')
#plt.plot(t_rms, (max(tonalness)/2)*vad_gt, label='VAD_gt')
#plt.grid()
#plt.xlabel('Time (s)')
#plt.legend(loc='best')
#plt.tight_layout()
#
#plt.show()

#%% 

ar = librosa.autocorrelate(audio, max_size = 1024)

#%%
plt.figure()
plt.plot(ar)



