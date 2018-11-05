# TRAIN INDIVIDUAL USERS

import pyaudio
import wave
import os
import librosa
import numpy as np
from hmmlearn.hmm import GMMHMM
import pickle
from sklearn.externals import joblib
import pandas as pd
import speech_recognition as sr
import re
from os import path

import pyaudio
import wave


import random
import librosa.display
import matplotlib.pyplot as plt
import cv2




# DATA AUGMENTATION AND COLLECTION 

training_speaker_name='shruthi'

file_path='C:/Anaconda codes/speaker reco/something new/for hack/dataset/'+training_speaker_name+'/'
file_names = os.listdir(file_path)
print((len(file_names)))

#speakers=["rahul","rajat","priti"]
speakers =["nishant","padma","rajat","shreekar","shruthi"]
#speakers =["nishant","rajat"]

for i in range(0,len(file_names)):
    speech, rate = librosa.core.load(file_path+file_names[i])
    print(rate)


lengths = np.empty(len(file_names))
print(np.shape(lengths))

feature_vectors = np.empty([20,0])

for i in range(len(file_names)):
    x, rate = librosa.load(file_path+file_names[i])               #loads the file
    #rate, x = wavfile.read(file_names[i])
    x=librosa.feature.mfcc(y=x[0:int(len(x)/1.25)], sr=rate)      #extracts mfcc
    
        
    #x = mfcc(x[0:len(x)/1.25], samplerate=rate)
    lengths[i] = int(len(x.transpose()))    
   
    print(np.shape(x))
    
    feature_vectors = np.concatenate((feature_vectors, x),axis=1)
    #feature_vectors = np.vstack((feature_vectors, x.transpose()))
    
print(((lengths)))
print(np.shape(feature_vectors))

#TRAINING A MODEL


N = 3  # Number of States of HMM
Mixtures = 64# Number of Gaussian Mixtures.


model = GMMHMM(n_components=N, n_mix=Mixtures, covariance_type='diag')

startprob = np.ones(N) * (10**(-30))  # Left to Right Model
startprob[0] = 1.0 - (N-1)*(10**(-30))
transmat = np.zeros([N, N])  # Initial Transmat for Left to Right Model
print(startprob,'\n',transmat)
for i in range(N):
    for j in range(N):
        transmat[i, j] = 1/(N-i)
transmat = np.triu(transmat, k=0)
transmat[transmat == 0] = (10**(-30))


model = GMMHMM(n_components=N, n_mix=Mixtures, covariance_type='diag', init_params="mcw",n_iter=100)

model.startprob_ = startprob
model.transmat_ = transmat
print(startprob,'\n',transmat)

feature=feature_vectors.transpose()
print(np.shape(feature))

lengths = [ int(x) for x in lengths ]
print(type(lengths[0]))

model.fit(feature,lengths)

joblib.dump(model, "C:/Anaconda codes/speaker reco/something new/for hack/models/"+training_speaker_name+".pkl")

