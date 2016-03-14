# -*- coding: utf-8 -*-

import librosa
import matplotlib.pyplot as plt

#audio_file = '../traditional_dataset/syrinx/fragments/syrinx_first_fragment_douglas_mono.wav'
#audio_file = '../traditional_dataset/syrinx/fragments/syrinx_second_fragment_dwyer_mono.wav'
#audio_file = '../traditional_dataset/syrinx/fragments/syrinx_third_fragment_rhodes_mono.wav'
#audio_file = '../traditional_dataset/syrinx/fragments/syrinx_fourth_fragment_bernold_mono.wav'
#audio_file = '../traditional_dataset/syrinx/fragments/syrinx_fifth_fragment_bourdin_mono.wav'

#audio_file = '../traditional_dataset/allemande/fragments/allemande_second_fragment_gerard_mono.wav'
#audio_file = '../traditional_dataset/allemande/fragments/allemande_first_fragment_nicolet_mono.wav'
#audio_file = '../traditional_dataset/allemande/fragments/allemande_third_fragment_rampal_mono.wav'
#audio_file = '../traditional_dataset/allemande/fragments/allemande_fourth_fragment_larrieu_mono.wav'
audio_file = '../traditional_dataset/allemande/fragments/allemande_fifth_fragment_preston_mono.wav'

audio, sr = librosa.load(audio_file, sr=44100, mono=True)

plt.figure()
plt.plot(audio)
#plt.tight_layout()

rms = librosa.feature.rmse(audio)

plt.figure()
plt.semilogy(rms.T, label='RMS Energy')
plt.xticks([])
plt.xlim([0, rms.shape[-1]])
plt.legend(loc='best')