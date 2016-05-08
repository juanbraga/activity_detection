# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def zero_crossing_rate(audio, fs=44100, n=1024):    
    
    result = np.empty(len(audio)/n)
    audio_sgn = np.sign(audio)    
    for i in range(0,len(audio)/n):
        result[i] = (np.sum(np.absolute(np.diff(audio_sgn[i*n:(i+1)*n])))/float(2*n))
    t = np.arange(len(result)) * (float(n)/fs) 
    result_f = (1000*float(fs)/(n))*result
    return result, result_f, t
    
    #%%

def average_energy(audio, fs=44100, n=1024):
    
<<<<<<< .merge_file_a12456
    result = np.empty(len(audio)/n)
    for i in range(0,len(audio)/n):
        result[i] = (np.sum(np.absolute(np.hamming(n)*audio[i*n:(i+1)*n]))/float(n))
=======
    Ew = np.sum(np.hamming(n)**2)
    result = np.empty(len(audio)/n)
    for i in range(0,len(audio)/n):
        result[i] = (np.sum(np.absolute(np.hamming(n)*audio[i*n:(i+1)*n]))/(float(n)*Ew))
>>>>>>> .merge_file_a12560
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

    import csv
    dataset=[]    
    cr = csv.reader(open('dataset.csv',"rb"))
    for row in cr:
        dataset.append(row[0]) 
    
    total_zcr_silence = np.empty([0,])
    total_ae_silence = np.empty([0,])
    total_mc_silence = np.empty([0,])
    total_zcr_activity = np.empty([0,])
    total_ae_activity = np.empty([0,])
    total_mc_activity = np.empty([0,])

    for i in range(0,len(dataset)):
    
        silence_file = dataset[i] + '_silence.wav'
        activity_file = dataset[i] + '_activity.wav'
        
        fs, audio_silence = wav.read(silence_file)
        fs, audio_activity = wav.read(activity_file)        
        
        frame_size=1024
        
        zcr_silence, t_zcr_silence = zero_crossing_rate(audio_silence, fs, n=frame_size)  
        zcr_activity, t_zcr_activity = zero_crossing_rate(audio_activity, fs, n=frame_size) 
        ae_silence, t_ae_silence = average_energy(audio_silence, fs, n=frame_size)  
        ae_activity, t_ae_activity = average_energy(audio_activity, fs, n=frame_size) 
        mc_silence, t_mc_silence, dummy = morph_close(audio_silence, fs, n=frame_size)  
        mc_activity, t_mc_activity, dummy = morph_close(audio_activity, fs, n=frame_size) 
        
        total_zcr_silence = np.r_[total_zcr_silence,zcr_silence]
        total_ae_silence = np.r_[total_ae_silence,ae_silence]
        total_mc_silence = np.r_[total_mc_silence,mc_silence]
        total_zcr_activity = np.r_[total_zcr_activity,zcr_activity]
        total_ae_activity = np.r_[total_ae_activity,ae_activity]
        total_mc_activity = np.r_[total_mc_activity,mc_activity]

#    plt.figure()    
#    plt.subplot(3,1,1)    
#    plt.plot(total_ae_activity)
#    plt.plot(total_ae_silence)
#    plt.grid()
#    plt.axis('tight')
#    plt.subplot(3,1,2)    
#    plt.plot(total_zcr_activity)
#    plt.plot(total_zcr_silence)    
#    plt.grid()
#    plt.axis('tight')    
#    plt.subplot(3,1,3)    
#    plt.plot(total_mc_activity)
#    plt.plot(total_mc_silence)
#    plt.grid()
#    plt.axis('tight')    

#%%
    plt.figure(figsize=(18, 6))    
    plt.subplot(3,1,1)    
    plt.hist([total_zcr_activity, total_zcr_silence], bins = 200)
    plt.axis('tight')
    plt.subplot(3,1,2)    
    plt.hist([total_ae_activity, total_ae_silence], bins = 200)
    plt.axis('tight')    
    plt.subplot(3,1,3)    
    plt.hist([total_mc_activity, total_mc_silence], bins = 200)
    
    plt.savefig('docs/hists.eps', format='eps', dpi=1000)
    
#%%
        
    plt.figure()
    plt.clf()
    target = np.r_[np.ones(len(total_ae_activity)), np.zeros(len(total_ae_silence))]
    plt.scatter(np.r_[total_ae_activity, total_ae_silence], np.r_[total_zcr_activity, total_zcr_silence], c=target)
    plt.legend(loc='best')    
    plt.xlabel('ae')
    plt.ylabel('zcr')
    plt.grid()
    plt.axis('tight')
    
    #%%
    plt.figure()
    plt.clf()
    plt.subplot(2,1,1)    
    plt.scatter(total_ae_silence,total_zcr_silence)
    plt.legend(loc='best')    
    plt.xlabel('ae')
    plt.xlim([0, 4000])
    plt.ylim([0, 1])
    plt.ylabel('zcr')
    plt.grid()
    plt.subplot(2,1,2)    
    plt.scatter(total_ae_activity, total_zcr_activity)
    plt.legend(loc='best')    
    plt.xlabel('ae')
    plt.ylim([0, 1])    
    plt.xlim([0, 4000])
    plt.ylabel('zcr')
    plt.grid()
    
    #%%
    # TRAIN
    from sklearn import svm
    ntest_activity = 900
    ntest_silence = 200
    y_train = np.r_[np.ones(len(total_ae_activity)-ntest_activity), np.zeros(len(total_ae_silence)-ntest_silence)]
    X_train = np.c_[np.r_[total_ae_activity[:-ntest_activity], total_ae_silence[:-ntest_silence]], np.r_[total_zcr_activity[:-ntest_activity], total_zcr_silence[:-ntest_silence]]] 
    y_test = np.r_[np.ones(ntest_activity), np.zeros(ntest_silence)]
    X_test = np.c_[np.r_[total_ae_activity[-ntest_activity:len(total_ae_activity)], total_ae_silence[-ntest_silence:len(total_ae_silence)]], np.r_[total_zcr_activity[-ntest_activity:len(total_zcr_activity)], total_zcr_silence[-ntest_silence:len(total_zcr_silence)]]]     
    svm_clf = svm.SVC(C=1.1111111111111112, cache_size=250007, class_weight=None, coef0=0.0,
        decision_function_shape='ovo', degree=1, gamma='auto', kernel='linear',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)      
    svm_clf.fit(X_train, y_train) 
        
    #%% TEST
    #svm_clf.predict(X_test)
    svm_clf.score(X_test, y_test)
    
    plt.figure()
    plt.clf()
    plt.scatter(X_test[:,0], X_test[:,1], c=y_test)
    plt.legend(loc='best')    
    plt.xlabel('ae')
    plt.ylabel('zcr')
    plt.grid()
    plt.axis('tight')
    

#%%

    from sklearn import tree 
    tree_clf = tree.DecisionTreeClassifier()
    tree_clf.fit(X_train, y_train) 
    
    # TEST
    #tree_clf.predict(X_test)
    tree_clf.score(X_test,y_test)

#%% FILTRATE
    import scipy.io.wavfile as wav
    
    Fs = 44100
    f = 1
    sample = 44100
    x = np.arange(sample)
    t = x/float(sample);
    y = np.sin(2 * np.pi * f * x / Fs)    
    
    fs, audio = wav.read("C:/Users/juan.braga/Desktop/audio_dataset/UrbanSound/data/siren/26173.wav")
    zcr, f0, t_zcr = zero_crossing_rate(y, fs=44100, n=22050)
    plt.figure()
    plt.subplot(2,1,1)    
    plt.plot(t,y)
    plt.grid()
    plt.subplot(2,1,2)    
    plt.plot(t_zcr, zcr)
    plt.grid()
