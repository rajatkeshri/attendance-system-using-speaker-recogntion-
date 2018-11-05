import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D , MaxPooling2D
from keras.utils import to_categorical
import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm


names=["nishant","rajat","shreekar","shruthi"]

#--------------------------------------
#EXTRACTING MFCC AND PADDING ZEROES THEN SAVING IT AS .NPY FILES

def wav2mfcc(path):
    max_len=13
    
    wave,sr=librosa.load(path,sr=None)
    #wave=wave[::3] #downsampling
    
    mfcc=librosa.feature.mfcc(wave,sr=16000)
    
    if (max_len>mfcc.shape[1]):
        pad_width=max_len-mfcc.shape[1]
        mfcc=np.pad(mfcc,pad_width=((0,0),(0,pad_width)),
                    mode='constant')
        
    else:
        mfcc=mfcc[:,:max_len]
    return mfcc
#---------------------------------------


#RECORD VOICE AND SAVE IT IN .WAV FILE


import pyaudio
import wave
import os
import librosa
import numpy as np
from hmmlearn.hmm import GMMHMM
import pickle
from sklearn.externals import joblib
 
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "C:/Anaconda codes/speaker reco/something new/samples/file_out.wav"
 
audio = pyaudio.PyAudio()
 
# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
print("recording...")
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

import tensorflow as tf
from keras.models import load_model

model = load_model("speaker_reco.model")
g="C:/Anaconda codes/speaker reco/something new/samples/file_out.wav"
#g='C:/Anaconda codes/digits speech recognition/free-spoken-digit-dataset-master/7.wav'
#sample=wav2mfcc('C:/Anaconda codes/digits speech recognition/free-spoken-digit-dataset-master/recordings/5/5_jackson_5.wav')

sample=wav2mfcc(g)
sample=sample.reshape(1,20,13,1)
#print(sample)

predictions=model.predict(sample)
c=np.argmax(model.predict(sample))

print(predictions)
print(c)

y=np.argmax(model.predict(sample))
p=names[y]

print(p)

import speech_recognition as sr

# obtain path to "Daily_English_Conversation_02_Do_you_speak_English.wav" in the same folder as this script
from os import path
AUDIO_FILE = (  "C:/Anaconda codes/speaker reco/something new/samples/file_out.wav")

# use the audio file as the audio source
r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
    #print("Say something!")
    audio = r.record(source)  # read the entire audio file
try:
    # for testing purposes, we're just using the default API key
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    # instead of `r.recognize_google(audio)`
    print("Google Speech Recognition thinks you said : " + r.recognize_google(audio))
    number=r.recognize_google(audio)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
    
number=int(number)
print(number)



if (p=="rajat" and number==127):
    print("attendance given to rajat")
elif (p=="nishant" and number==102):
    print("attendance given to nishanth")
else:
    print("dont try to give proxy")
