import speech_recognition as sr
import time
import numpy as np

training_index = 1

AUDIO_FILE = "training_"+str(training_index)+".mp3"
print(AUDIO_FILE)

import subprocess

subprocess.call(['ffmpeg', '-i', AUDIO_FILE,'audio.wav'], shell=True)
AUDIO_FILE = 'audio.wav'

'''
file = open('training_label.txt')
  
# read the content of the file opened
content = file.readlines()
  
# read training_index th line from the file
Label = content[int(training_index-1)]

print("Label: ", Label)
'''

##Below is the implementation from Google Speech Recognition

# use the audio file as the audio source
r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
    audio = r.record(source)  # read the entire audio file



# recognize speech using Google Speech Recognition
try:
    print("Google recognized sentence: ", r.recognize_google(audio))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))