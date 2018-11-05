#NEW PERSON
#RUN NEWTRAIN FUNCTION TO ADD NEW STUDENT



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
import speech_recognition as sr
import re
from os import path

import pyaudio
import wave
import os
import librosa
import numpy as np
from hmmlearn.hmm import GMMHMM
import pickle
from sklearn.externals import joblib
import wave
import csv


#-----------------------------------------------------------------
def samples(path):
        for i in range(0,5):
            #RECORD SPEAKERS VOICE FOR GIVING ATTENDANCE

            FORMAT = pyaudio.paInt16
            CHANNELS = 2
            RATE = 44100
            CHUNK = 1024
            RECORD_SECONDS = 3
            WAVE_OUTPUT_FILENAME = path+"file"+str(i)+".wav"

            audio = pyaudio.PyAudio()

            # start Recording
            stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
            print(str(i)+" recording...")
            frames = []

            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            print("finished recording")

             # stop Recording
            stream.stop_stream()
            stream.close()
            audio.terminate()

            waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()

#---------------------------------------------------------------
def newtrain(speakers):
    #folder="C:/Anaconda codes/speaker reco/something new/for hack/add new people/"
    folder="C:/Anaconda codes/speaker reco/something new/for hack/"

    l=len(speakers)
    name= input("enter your name")

    speakers.append(name)

    new_person=speakers[l]

    #rint(new_person)
    os.mkdir(folder+"/dataset/"+ name)


    x=folder+"/dataset/"+name+"/"
    samples(x)

    training_speaker_name=name

    file_path=x
    file_names = os.listdir(file_path)
    print((len(file_names)))


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

    joblib.dump(model, folder+"/models/"+name+".pkl")
    return(speakers)
    
speakers =["nishant","padma","rajat","shreekar","shruthi"]
print("enter 1 to train new user,else give attendance")

i1=input()

if i1==str(1):
    speakers1=newtrain(speakers)
    with open('s.txt', 'w') as f:
        for item in speakers1:
            f.write("%s\n" % item)
else:
    exit()

