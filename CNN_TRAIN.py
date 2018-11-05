import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm


names=["nishant","rajat","shreekar","shruthi"]

#LOADING THE DATA
datapath="C:/Anaconda codes/speaker reco/something new/for hack/dataset2/"

labels=os.listdir(datapath)
print(labels)

label_indices=np.arange(0,len(labels))
print(label_indices)

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
    



save_numpy='C:/Anaconda codes/speaker reco/something new/for hack/numpy/'

for i in labels:
    print(i)
    mfcc_vect=[]

    j = os.listdir(datapath+i+'/')
    #print(len(j))
    
    for k in range(0,len(j)):
        #print(datapath+i+'/'+j[k])
        mfcc=wav2mfcc(datapath+i+'/'+j[k])
        print(np.shape(mfcc))
        mfcc_vect.append(mfcc)
    #print(np.shape(mfcc_vect))
    np.save('C:/Anaconda codes/speaker reco/something new/for hack/numpy/' 
            +i+ '.npy',mfcc_vect)
    #print('done')


#SPLIT DATA FOR TRAIN AND TEST

def get_train_test(split_ratio=0.95, random_state=40):
    # Get available labels

    # Getting first arrays
    X = np.load(save_numpy+labels[0] + '.npy')
    y = np.zeros(X.shape[0])
    
    #print(X,y)
    
    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(save_numpy+label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D , MaxPooling2D
from keras.utils import to_categorical


X_train, X_test,y_train,y_test=get_train_test()

max_len=13

X_train= X_train.reshape(X_train.shape[0],20,max_len,1)
X_test= X_test.reshape(X_test.shape[0],20,max_len,1)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

print(np.shape(X_train))

model=Sequential()

model.add(Conv2D(32, kernel_size=(2,2), activation="relu",
          input_shape=(20,max_len,1)))
model.add(Conv2D(64,kernel_size=(2,2),activation="relu"))
model.add(Conv2D(128,kernel_size=(2,2),activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(128,activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(4,activation="softmax"))

model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train,y_train_hot,batch_size=10,epochs=25,verbose=1,
         validation_data=(X_test,y_test_hot))

model.save("speaker_reco.model")
