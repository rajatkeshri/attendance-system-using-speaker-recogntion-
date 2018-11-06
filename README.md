Attendance system using speaker recogniton and speech recognition to avoid proxy .


We used two approaches here - HIDDEN MARKOV MODEL APPROACH AND CNN MODEL APPROACH 

#-------------------------------------------------------------------------------------------------------------------------------------
HIDDEN MARKOV MODEL APPROACH 
Inside the Hidden markov model folder there are multiple files

The NEW_TRAIN.py file is used to train a new user. It takes in 5 voice samples from the user and trains on it and then stores the voice model in the models folder.
The training data is stored in a folder inside the dataset, with the speaker name as folder name.

The INDIVIDUAL_TRAIN.py is a script which trains speaker's voice, provided we already have the training data stored in a folder in the datasets folder. 
This script trains the model and stores it in the models folder.

The ONLY_PREDICT.py file has the script which takes in a new sample value from the speaker , and  predicts who the speaker is. The speaker is asked to speak his/her roll number into the microphone and googles speech to text api converts it into text. If the roll number matches the roll number of the given person in the database, then a random word will be generated which the speaker has to speak again , for higher authentication, to prevent proxy givers if they plan to use a recorder audio of some other speaker.
After this process, attendance will be given and the database will be updated.

RESET_DB.py is a file which is used to reset the  attendance coloumn in .csv attendance file to all zeroes for the next day's attendance

update_to_db.py is a file which updates the attendance to the database.

s.txt holds the spakers names.
Registry holds the attendance of all the days and keeps getting updated.

#--------------------------------------------------------------------------------------------------------------------------------------

Inside the folder front end and backend, there is a script, which has all the functions which is used to predict and give attendance.
It also include a front end in that same script.
Just run this script after training and properly specifying the path for everything, and it will ask for your voice input and predict who the speaker is among the lost of speakers.

#--------------------------------------------------------------------------------------------------------------------------------------

REMEMBER TO CHANGE THE PATH FOR MODELS AND DATASETS BECAUSE MY PATH USED WILL DEFINATELY BE DIFFERENT FROM THE PATH YOU USE.


#-------------------------------------------------------------------------------------------------------------------------------------

CNN APPROACH
Run the CNN_TRAIN.py file to run training on your datasets. Put in your training samples in the folder called DATASETS, inside which create a folder with each speakers name and put his/her voices in that particular folder

Run CNN_PREDICT.py to accept a new spoken voice which is recorded when you run the code and it then predicts who the speaker is.
