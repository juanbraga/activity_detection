# -*- coding: utf-8 -*-

import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
from sklearn import (manifold, decomposition, datasets, ensemble,
                     discriminant_analysis, random_projection)

if __name__ == "__main__":  

    melbands = 128
    winlen = 2048
    hop = winlen/4

    import csv
    dataset=[]    
    cr = csv.reader(open('dataset.csv',"rb"))
    for row in cr:
        dataset.append(row[0]) 
    
    total_mfcc_silence = np.empty([melbands,])
    total_mfcc_activity = np.empty([melbands,])

    for i in range(0,2):#len(dataset)):
    
        silence_file = dataset[i] + '_silence.wav'
        activity_file = dataset[i] + '_activity.wav'
        
        fs, audio_silence = wav.read(silence_file)
        fs, audio_activity = wav.read(activity_file)        
        
        frame_size=1024

        mfccs_silence = librosa.feature.mfcc(y=audio_silence, sr=fs, n_mfcc=melbands,
                                             n_mels=melbands, n_fft=winlen, hop_length=hop)
        mfccs_activity = librosa.feature.mfcc(y=audio_activity, sr=fs, n_mfcc=melbands,
                                              n_mels=melbands, n_fft=winlen, hop_length=hop)        
        
        total_mfcc_silence = np.c_[total_mfcc_silence,mfccs_silence]
        total_mfcc_activity = np.c_[total_mfcc_activity,mfccs_activity]
        
    plt.figure()    
    librosa.display.specshow(total_mfcc_activity, x_axis='time')
    plt.colorbar()
    plt.title('MFCC activity')
    plt.tight_layout()
    
    plt.figure()
    librosa.display.specshow(total_mfcc_silence, x_axis='time')
    plt.colorbar()
    plt.title('MFCC silence')
    plt.tight_layout()
    
#    # Projection on to the first 2 linear discriminant components
#    print("Computing Linear Discriminant Analysis projection")
#    X2 = X.copy()
#    X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
#    t0 = time()
#    X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=3).fit_transform(X2.transpose(), y)
#    #plot_embedding(X_lda, y,
#    #               "Linear Discriminant projection of the digits (time %.2fs)" %
#    #               (time() - t0))
#    fig1 = plt.figure(1, figsize=(16, 9))
#    ax1 = fig1.add_subplot(111, projection='3d')
#    ax1.scatter(X_lda[:,0], X_lda[:,1], X_lda[:,2], c=color)