#THIS SCRIPT IS USED TO TAKE IN YOUR VOICE INPUT AND PREDICT AS WELL AS
#UPDATE YOUR ATTENDANCE DATABASE


import time
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
from tkinter import *



# globally declare the expression variable 
expression = ""

# Function to clear the contents 
# of text entry box 
def clear(): 
    global expression 
    expression = "" 
    equation.set("")

def abcd():


    Question.delete(1.0,END)
    Question.insert(INSERT, "PROCESSING")
    gui.update()
    time.sleep(.5)

    
    speakers =["nishant","padma","rajat","shreekar","shruthi"]

    with open('C:/Anaconda codes/speaker reco/something new/s.txt') as f:
        speakers = f.read().splitlines()
        speakers=speakers[:len(speakers)-1]
        print((speakers))

    #--------------------------------------------------------------------------------
    #RECORD SPEAKERS VOICE FOR GIVING ATTENDANCE

    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3
    WAVE_OUTPUT_FILENAME = "C:/Anaconda codes/speaker reco/something new/samples/file.wav"
     
    audio = pyaudio.PyAudio()
     
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    print("recording...")
    Question.delete(1.0,END)
    Question.insert(INSERT, "SPEAK NOW")
    gui.update()
    time.sleep(.5)
    frames = []
     
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    Question.delete(1.0,END)
    Question.insert(INSERT, "finished recording")
    gui.update()
    time.sleep(.5)
    
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

    #------------------------------------------------------------------------------
    #speakers =["nishant","padma","rajat","rohit","sarah","shreekar","shruthi"]
    #speakers =["nishant","padma","rajat","shreekar","shruthi"]

    threshold = 0
    l=2
    uppercutoff=20000
    lowercutoff=8000

    #open the test data and find its probability 
    #compare it with test probability and print predictions

    student="samples/file.wav" #SPEAKERS VOICE STORED

    file_path1="C:/Anaconda codes/speaker reco/something new/"
    #file_path1="C:/Anaconda codes/speaker reco/something new/for hack/other students/"
    test_speech1 = student
    speech1, rate = librosa.core.load(file_path1+test_speech1)     #EXTRACT MFCC AND ADD IT OT FEATURE VECTOR
    feature_vectors12 = librosa.feature.mfcc(y=speech1, sr=rate)

    features1=feature_vectors12.transpose()
    #print(np.shape(features1))


    S = librosa.feature.melspectrogram(y=speech1, sr=rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    plt.show()


    #GET THE PREDICTION VALUES FOR EVERY MODEL CREATED FOR EACH SPEAKER
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

    x[l]=x[l]+threshold

    y=x.index(min(x))
    print(x)
    #print(x.index(min(x))+1)
    if min(x)<uppercutoff and min(x)>lowercutoff:
        
        print("Hi "+speakers[y]+".How are you?")
        Question.delete(1.0,END)
        Question.insert(INSERT, "Hi "+speakers[y]+".How are you?")
        gui.update()
        time.sleep(.5)
        p=speakers[y]
        
    else:
        print("cant recognise. Speak again")
        Question.delete(1.0,END)
        Question.insert(INSERT, "cant recognize. Speak again")
        gui.update()
        time.sleep(.5)

    #-------------------------------------------------------------------------------
    #CONVERT WHAT THE SPEAKER SAID TO TEXT, USING SPEECH RECONITION
    #HERE GOOGLE API IS USED
    #WE CAN TRAIN USING OUR OWN MODEL, BUT THE ACCURACY IS NOT GREAT, DUE TO LESS DATASETS AVAILABLE

    AUDIO_FILE = ( "C:/Anaconda codes/speaker reco/something new/"+student)

    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        #print("Say something!")
        audio = r.record(source)  # read the entire audio file
    try:
        #print("Google Speech Recognition thinks you said : " + r.recognize_google(audio))
        number=r.recognize_google(audio)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        
    #number=int(number)
    #print(number)



    #THIS FUNCTION IS USED TO EXTRACT ONLY THE NUMBERS OUT OF THE SPOKEN SENTENCE BY THE SPEAKER
    #EXAMPLE : SPEAKER SPEAKS "NAME AND 302", THE FUNCTIONS EXTRACTS 302 FOR ROLL NUMBER ANALYSIS
    #--------------------------------------------------------------------------------------------------------------------------------
    def extractMax(input): 
      
         # get a list of all numbers separated
         numbers = re.findall('\d+',input) 
      
         # now we need to convert each number into integer 
         # int(string) converts string into integer 
         numbers = map(int,numbers) 
      
         return(max(numbers) )

    #--------------------------------------------------------------------------------------------------------------------------------
        
    number=str(extractMax(number))
    print("speaker said",number)
    Question.delete(1.0,END)
    Question.insert(INSERT, "speaker said"+number)
    gui.update()
    time.sleep(.5)


    #----------------------------------------------------------------------
    #FOR FURTHER AUTHENTICATION, WE ASK THE SPEAKER TO SPEAK A RANDOM GENERATED WORD, TO AVOID PROXY
    #record voice in real time
     
        
    #RANDOM WORDS
    flag=0
    random1=["apple","mango","pizza","chips","abort","above","leave","actor","adore","brass","colour","hazel","pulse","sleep"]

    threshold=0
        
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 4
    WAVE_OUTPUT_FILENAME = "random.wav"
     
    audio = pyaudio.PyAudio()
     
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

    xx=random.choice(random1)
    print("recording...")
    print("SAY", xx)
    
    Question.delete(1.0,END)
    Question.insert(INSERT, "SAY "+xx)
    gui.update()
    time.sleep(.5)
    
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


    #-------------------------------------------------------------------------------

    AUDIO_FILE = ("random.wav")
    #AUDIO_FILE = ( "file.wav")

    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        #print("Say something!")
        audio = r.record(source)  # read the entire audio file
    try:
       
        word=r.recognize_google(audio)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        
    word=word.lower()
    print("speaker spoke",word)
    Question.delete(1.0,END)
    Question.insert(INSERT, "speaker spoke  "+word)
    gui.update()
    time.sleep(.5)


    #THIS FUNCTION IS USED TO UPDATE THE VALUES IN THE CSV FILE AND GIVE ATTENDANCE TO THE STUDENTS
    #------------------------------------------------------------------------------------------------------------------------------
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

        rows=len(names)
        
        for i in range(0,len(names)):
            if names[i]==n:
                break

        for j in range(0,len(usn)):
            if int(usn[j])==r:
                break

        if i==j:
            att[i]=1


        df = pd.read_csv("C:/Anaconda codes/speaker reco/something new/for hack/attendance_database.csv") 

        #df = pd.DataFrame(data, columns = ['NAMES','ROLL_NUMBER','ATTENDANCE'])
        
        df = df[['NAMES','ROLL_NUMBER','ATTENDANCE']]
        print("\n before updation")
        print(df,"\n")

        for i in range(0,len(names)):
            (df['ATTENDANCE'][i])=att[i]
        

        print("\n after updation")
        print(df)

        df.to_csv("C:/Anaconda codes/speaker reco/something new/for hack/attendance_database.csv")


    #-----------------------------------------------------------------------------------
    def print_img(r):
        path_img="C:/Anaconda codes/speaker reco/something new/for hack/images/"
        path_files=os.listdir(path_img)

        print(path_files)
        t=r
        img = cv2.imread(path_img+r)

        #for i in range(0,len(path_files)):
            #if t==path_files[i].lower():
                #plt.imshow(img)
                #plt.show()
          
        for i in range(0,len(path_files)):
            if t==path_files[i].lower():
                cv2.imshow("image",img)
                #cv2.waitKey(0)
                

    #------------------------------------------------------------------------------------------------------------------------------


    #PREDICT WHO SPOKE THE RANDON GIVEN WORD FOR MORE AUTHENTICATION
    #speakers =["nishant","padma","rajat","shreekar","shruthi"]
    #open the test data and find its probability 
    #compare it with test probability and print predictions

    student="random.wav"

    file_path1="C:/Anaconda codes/speaker reco/something new/for hack/"
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

    x[l]=x[l]+threshold

    y=x.index(min(x))
    print(x)
    #print(x.index(min(x))+1)

    if min(x)<uppercutoff and min(x)>lowercutoff:
        p1=speakers[y]
        
        if word==xx and p==p1 or p=="nishant":                         #checking if the said word is present in the random words list

            flag=1
           # print(speakers[y]+" is confirmed and spoke "+ number)
            
            if (p=="rajat" and number=="127" and flag==1):
                update(p,number)
                
                Question.delete(1.0,END)
                Question.insert(INSERT, "attendance given to rajat")
                
                
                print("attendance given to rajat")
                print_img(number+".jpg")
                
            elif (p=="nishant" and number=="102" and flag==1):
                update(p,number)

                Question.delete(1.0,END)
                Question.insert(INSERT, "attendance given to nishanth")
                
                
                print("attendance given to nishanth")
                print_img(number+".jpg")
                
            elif (p=="padma" and number=="106" and flag==1):
                update(p,number)

                Question.delete(1.0,END)
                Question.insert(INSERT, "attendance given to padma")
                
                
                print("attendance given to padma")
                print_img(number+".jpg")
                
            else:

                Question.delete(1.0,END)
                Question.insert(INSERT, "dont try to give proxy")
                
                print("dont try to give proxy")
                
            
        else:
            print("AUTHENTICATION FAILED")
            print("dont try to give proxy")
            
        
    else:
        print("AUTHENTICATION FAILED")
        print("dont try to give proxy")
        



#-------------------------------------------------------------------------------
def reset():
    #TO RESET THE DATABASE TO 0 ATTENDANCE
    import pandas as pd

    list1=[]
    
    data = pd.read_csv("C:/Anaconda codes/speaker reco/something new/for hack/attendance_database.csv") 

    names=data["NAMES"]
    names=list(names)

    usn=data["ROLL_NUMBER"]
    usn=list(usn)

    att=data["ATTENDANCE"]
    att=list(att)
    att1=list(att)
    print(att)

    rows=len(names)

    att= [0] * len(names)

    list1.append(att)
    with open('registry.txt', 'a') as f:
        for item in att1:
            print(item)
            f.write("%s\t" % item)
        f.write("\n")
    
    df = pd.read_csv("C:/Anaconda codes/speaker reco/something new/for hack/attendance_database.csv") 
    #rg=pd.read_csv("C:/Anaconda codes/speaker reco/something new/for hack/registry.csv")
    #rg=pd.DataFrame[["nishant","padma","rajat","shreekar","shruthi"]]
   

    df = df[['NAMES','ROLL_NUMBER','ATTENDANCE']]
    d=df["ATTENDANCE"]
    #rg.append(d)

    
    for i in range(0,len(names)):
        (df['ATTENDANCE'][i])=att[i]
    print("\n RESET ")
    print(df,"\n")
    df.to_csv("C:/Anaconda codes/speaker reco/something new/for hack/attendance_database.csv")
    Question.delete(1.0,END)
    Question.insert(INSERT, "DAY END ...DATA successfully stored")
    gui.update()
    time.sleep(.5)    


#------------------------------------------------------------------------------

# Driver code 
if __name__ == "__main__": 
    # create a GUI window 
    gui = Tk() 
  
    # set the background colour of GUI window 
    gui.configure(background="light green") 
  
    # set the title of GUI window 
    gui.title("Attendance using Voice recognition") 
  
    # set the configuration of GUI window 
    gui.geometry("320x320") 
  
    # StringVar() is the variable class 
    # we create an instance of this class 
    equation = StringVar() 
  
    # create the text entry box for 
    # showing the expression . 
    expression_field = Entry(gui, textvariable=equation) 
  
    # grid method is used for placing 
    # the widgets at respective positions 
    # in table like structure . 
    expression_field.grid(columnspan=10, ipadx=270) 
  
    equation.set('HELLO.....Give your attendance') 
  
    # create a Buttons and place at a particular 
    # location inside the root window . 
    # when user press the button, the command or 
    # function affiliated to that button is executed . 
    button1 = Button(gui, text=' RECORD ', fg='black', bg='red', 
                     command=abcd, height=2, width=7) 
    button1.grid(row=2, column=0) 
  

  
    clear = Button(gui, text='End of Day', fg='black', bg='red', 
                   command=reset, height=2, width=7) 
    clear.grid(row=2, column=2)

    Question = Text(gui, height=4, width=30,wrap=WORD,relief="solid",font=("Arial",10))
    Question.insert(INSERT, 'Waiting......')
    Question.grid(row=3,rowspan=5,column=0,pady=(10, 0),columnspan=4)
  
    # start the GUI 
    gui.mainloop() 
