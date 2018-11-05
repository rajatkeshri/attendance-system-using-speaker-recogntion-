import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm

import re

import pandas as pd
import pyaudio
import wave
import os
import librosa
import numpy as np
from hmmlearn.hmm import GMMHMM
import pickle
from sklearn.externals import joblib
import tensorflow as tf
from keras.models import load_model





#----------------------------RECORD---------------------------------------------
 
#record voice in real time
 
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "file.wav"
 
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

#----------------------------------------------------------------------------


speakers =["nishant","padma","rajat"]

#m1=joblib.load("C:/Anaconda codes/speaker reco/something new/models/rajat.pkl")  #load the model-
#m2=joblib.load("C:/Anaconda codes/speaker reco/something new/models/nishant.pkl") 
#m3=joblib.load("C:/Anaconda codes/speaker reco/something new/models/padma.pkl")#load the model

student="file.wav"

#open the test data and find its probability 
#compare it with test probability and print predictions

file_path1="C:/Anaconda codes/speaker reco/something new/"
test_speech1 = student
speech1, rate = librosa.core.load(file_path1+test_speech1)
feature_vectors12 = librosa.feature.mfcc(y=speech1, sr=rate)

features1=feature_vectors12.transpose()
#print(np.shape(features1))

x=[]

path ="C:/Anaconda codes/speaker reco/something new/for hack/models/"
names = os.listdir(path)
#print(names)

h=[]
for i in range(0,len(names)):
    m1=joblib.load("C:/Anaconda codes/speaker reco/something new/for hack/models/"+str(names[i]) )
    p1 = m1.score(features1)
    p1=abs(p1)
    x.append(p1)
    #print(m1.predict(features1),'\n')

"""
p2 = m2.score(features1)
p2=abs(p2)

p3 = m3.score(features1)
p3=abs(p3)

x.append(p1)
x.append(p2)
x.append(p3)
"""
y=x.index(min(x))
#print(x)
#print(x.index(min(x))+1)
if min(x)<20000 and min(x)>11000:
    
    print("Hi "+speakers[y]+".How are you?")
    p=speakers[y]
    
else:
    print("cant recognise. Speak again")
    exit()


#--------------------------------------------------------------------------------


import speech_recognition as sr

# obtain path to "Daily_English_Conversation_02_Do_you_speak_English.wav" in the same folder as this script
from os import path
AUDIO_FILE = ( student)

# use the audio file as the audio source
r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
    #print("Say something!")
    audio = r.record(source)  # read the entire audio file
try:
    # for testing purposes, we're just using the default API key
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    # instead of `r.recognize_google(audio)`
    #print("Google Speech Recognition thinks you said : " + r.recognize_google(audio))
    number=r.recognize_google(audio)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
    exit()
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
    exit()
#number=int(number)
#print(number)


def extractMax(input): 
     # get a list of all numbers separated by  
     # lower case characters  
     # \d+ is a regular expression which means 
     # one or more digit 
     # output will be like ['100','564','365'] 
     numbers = re.findall('\d+',input) 
  
     # now we need to convert each number into integer 
     # int(string) converts string into integer 
     # we will map int() function onto all elements  
     # of numbers list 
     numbers = map(int,numbers) 
     if max(numbers)==0:
         return false
    else:
         return(max(numbers) )
    
    
number=str(extractMax(number))
#print(number)

#------------------------------------------------------------------------------------


def update(n,r):
    r=int(r)
    data = pd.read_csv("C:/Anaconda codes/speaker reco/something new/for hack/attendance_database.csv") 

    names=data["NAMES"]
    names=list(names)

    usn=data["ROLL_NUMBER"]
    usn=list(usn)

    att=data["ATTENDANCE"]
    att=list(att)
    att1=list(att)

    #print(names)
    #print(usn)


    rows=len(names)


    #ASSUME NAME= RAJAT ROLL=127
    #n="nishant"
    #r=102

    for i in range(0,len(names)):
        #print(names[i])
        if names[i]==n:
            break

    #print(i)
    for j in range(0,len(usn)):
        #print(usn[j])
        if int(usn[j])==r:
            break

    if i==j:
        att[i]=1

    #print(att)



    df = pd.read_csv("C:/Anaconda codes/speaker reco/something new/for hack/attendance_database.csv") 

    #df = pd.DataFrame(data, columns = ['NAMES','ROLL_NUMBER','ATTENDANCE'])
    
    df = df[['NAMES','ROLL_NUMBER','ATTENDANCE']]

    print("\nBEFORE GIVING ATTENDANCE")
    print(df,"\n")


    for i in range(0,len(names)):
        (df['ATTENDANCE'][i])=att[i]
    

    #print(df['ATTENDANCE'])
    print("\nAFTER GIVING ATTENDANCE")
    print(df)

    df.to_csv("C:/Anaconda codes/speaker reco/something new/for hack/attendance_database.csv")
#---------------------------------------------------------------------------------
    
if (p=="rajat" and number=="127"):
    update(p,number)
    print("attendance given to rajat")
elif (p=="nishant" and number=="102"):
    update(p,number)
    print("attendance given to nishanth")
elif (p=="padma" and number=="106"):
    update(p,number)
    print("attendance given to padma")
else:
    print("dont try to give proxy")
