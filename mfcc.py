# -*- coding: utf-8 -*-

import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
from sklearn import (manifold, decomposition, datasets, ensemble,
                     discriminant_analysis, random_projection)
from sklearn.decomposition import PCA

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
    total_mfcc = np.empty([melbands,])
    labels = np.empty([1,], dtype='int16')

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
        
        aux_labels = np.r_[np.ones(len(mfccs_activity.T), dtype='int16'),np.zeros(len(mfccs_silence.T),dtype='int16')]
        labels = np.r_[aux_labels, labels]
        aux_total_mfcc = np.c_[mfccs_activity,mfccs_silence]
        total_mfcc = np.c_[aux_total_mfcc,total_mfcc]        
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

    total_mfcc = total_mfcc.T    
    
#%%
    from mpl_toolkits.mplot3d import Axes3D
    print("Computing Isomap 2dim")
    X_iso = manifold.Isomap(30, n_components=2).fit_transform(total_mfcc[:-1,:])
#%%    
    plt.figure(1, figsize=(8, 6))
    plt.clf()
    # Plot the training points
    plt.scatter(X_iso[:, 0], X_iso[:, 1], c=labels[:-1], cmap=plt.cm.Paired)
    plt.xlabel('1st')
    plt.ylabel('2nd')
    plt.show()

#%%
    print("Computing Isomap 3dim")
    X_iso_3 = manifold.Isomap(30, n_components=3).fit_transform(total_mfcc[:-1,:])

    fig = plt.figure(2, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter( X_iso_3[:, 0],  X_iso_3[:, 1],  X_iso_3[:, 2], c=labels[:-1],
               cmap=plt.cm.Paired)
    ax.set_title("Isomap 3dim")
    ax.set_xlabel("1st")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd")
    ax.w_zaxis.set_ticklabels([])
    
    plt.show()
  
    fig = plt.figure(3, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    print("Computing PCA")    
    X_reduced = PCA(n_components=3).fit_transform(total_mfcc[:-1,:])
    ax.scatter( X_reduced[:, 0],  X_reduced[:, 1],  X_reduced[:, 2], c=labels[:-1],
               cmap=plt.cm.Paired)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    
    plt.show()