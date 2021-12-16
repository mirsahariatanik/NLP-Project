import time
import numpy as np

#Speech generation reference: https://www.geeksforgeeks.org/convert-text-speech-python/

# Import the required module for text 
# to speech conversion
from gtts import gTTS
  
# This module is imported so that we can 
# play the converted audio
import os
import random

def generate_event(event_set,participant_set,week_set,day_set,hour_set,ampm_set):
    event = random.choice(event_set)
    participant = random.choice(participant_set)
    week = random.choice(week_set)
    day = random.choice(day_set)
    hour = random.choice(hour_set)
    ampm = random.choice(ampm_set)
    dates = random.choice(date_set)
    e= event.split()
    #print(event.split())
    #print(e[0])
    if (hour=='12'):
        event_generated = 'set'+' '+'reminder'+ ' '+ 'for' +' '+event + ' ' + 'with'  + ' ' + participant + ' ' + 'on the' + ' ' + dates + ' ' + 'from' + ' ' + hour+ampm+' '+'to'+ ' '+ '1'+ampm
        if (len(e)==1):
            eve= 'reminder:O'+' '+'set:O'+' '+'reminder:O'+' '+'for:O'+' '+event+ ':B-title'+ ' '+'with' + ':I-title'+' '+participant+':I-title'+' '+'on:O'+' '+'the:O'+' '+dates+':B-date' + ' '+ 'from'+':O'+' '+ hour+ampm+':B-start-time'+' '+'to'+':O'+' '+ '1'+ampm+':B-end-time'
        else:
            eve= 'reminder:O'+' '+'set:O'+' '+'reminder:O'+' '+'for:O'+' '+e[0]+ ':B-title'+ ' '+e[1]+ ':I-title'+' '+'with' + ':I-title'+' '+participant+':I-title'+' '+'on:O'+' '+'the:O'+' '+dates+':B-date' + ' '+ 'from'+':O'+' '+ hour+ampm+':B-start-time'+' '+'to'+':O'+' '+ '1'+ampm+':B-end-time'
        #find_taxi ( taxi-destination = backstreet bistro ; taxi-arriveby = 19:30 )
        frame= 'reminder' +' '+'('+' '+'title'+' '+'=' + ' '+ event+' '+'with'+ ' '+participant+' '+';'+ ' '+'date'+' '+ '='+' '+ dates+' '+';'+' '+'start-time'+' '+'='+' '+hour+ampm+' '+';'+' '+'end-time'+' '+'='+' '+'1'+ampm+' '+')'
    else:
        event_generated = 'set'+' '+'reminder'+ ' '+ 'for' +' '+event + ' ' + 'with'  + ' ' + participant + ' ' + 'on the' + ' ' + dates + ' ' +'from' + ' ' + hour+ampm+' '+'to'+ ' '+ str(int(hour)+1)+ampm
        if (len(e)==1):
            eve= 'reminder:O'+' '+'set:O'+' '+'reminder:O'+' '+'for:O'+' '+event+ ':B-title'+ ' '+'with' + ':I-title'+' '+participant+':I-title'+' '+'on:O'+' '+'the:O'+' '+dates+':B-date' +' '+ 'from'+':O'+' '+ hour+ampm+':B-start-time'+' '+'to'+':O'+' '+ str(int(hour)+1)+ampm+':B-end-time'
        else:
            eve= 'reminder:O'+' '+'set:O'+' '+'reminder:O'+' '+'for:O'+' '+e[0]+ ':B-title'+ ' '+e[1]+ ':I-title'+' '+'with' + ':I-title'+' '+participant+':I-title'+' '+'on:O'+' '+'the:O'+' '+dates+':B-date' +' '+ 'from'+':O'+' '+ hour+ampm+':B-start-time'+' '+'to'+':O'+' '+ str(int(hour)+1)+ampm+':B-end-time'
        #find_taxi ( taxi-destination = backstreet bistro ; taxi-arriveby = 19:30 )
        frame= 'reminder' +' '+'('+' '+'title'+' '+'=' + ' '+ event+' '+'with'+ ' '+participant+' '+';'+ ' '+'date'+' '+ '='+' '+ dates+' '+';'+' '+'start-time'+' '+'='+' '+hour+ampm+' '+';'+' '+'end-time'+' '+'='+' '+str(int(hour)+1)+ampm+' '+')'
    event_frame= event_generated+ '\t'+ frame
    event_bio= eve
    data_act= event_generated+'\t' + 'reminder'
    act= 'reminder'

    return event_generated, frame, event_frame, event_bio, data_act,act
    

event_set = ['meeting','talk','breakfast','lunch','dinner','conference','zoom','trip','flight', 'yoga activity', 'interview',
                'lab appointment',  'playing football', 'dinner', 'taking medicine', 'dentist appointment', 'meeting', 'football activities', 
                'optometrist appointment']
participant_set = ['John','Robert','James','Michael','William','David','Richard','Mary','Jennifer',
                   'Linda','Elizabeth','Barbara','Sarah','Jessica','Ashley','Emily','Lisa','Alice',
                   'Emily','Kevin','Brian','George','Timothy','Jeffrey','','Ryan','Jacob','Gary','Justin',
                   'Scott', 'Alex', 'father','sister', 'Liam',  'Olivia', 'Noah', 'Emma','Oliver', 'Ava',
                   'Elijah',  'Charlotte', 'William', 'Sophia','James', 'Amelia','Benjamin', 'Isabella','Lucas', 'Mia','Henry',
                    'Evelyn', 'Alexander', 'Harper']
day_set = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
week_set = ['this','next']
#hour_set = ['1pm','2pm','3pm','4pm','5pm','6pm','7pm','8pm','9pm','10pm','11am','12pm', '7am','8am','9am','10am','11am']
hour_set = ['1','2','3','4','5','6','7','8','9','10','11','12']
ampm_set = ['am','pm']
date_set= ['1st','2nd','3rd', '4th', '5th', '6th','7th','8th','9th','10th','11th', '12th','13th','14th','15th','16th','17th','18th','19th','20th','21st', '22nd','23rd', '24th','25th','26th','27th','28th','29th','30th','31st']


file_index = 101

training_count = 100
for i in range(training_count):
    
    # The text that you want to convert to audio
    event_generated, frame, event_frame, event_bio, data_act, act = generate_event(event_set,participant_set,week_set,day_set,hour_set,ampm_set)

    # Language in which you want to convert
    language = 'en'
      
    # Passing the text and language to the engine, 
    # here we have marked slow=False. Which tells 
    # the module that the converted audio should 
    # have a high speed
    '''
    myobj1 = gTTS(text=event_generated, lang=language, slow=False)
    myobj2 = gTTS(text=frame, lang=language, slow=False)
    myobj3 = gTTS(text=event_frame, lang=language, slow=False)
    myobj4 = gTTS(text=event_bio, lang=language, slow=False)
    '''  
    # Saving the converted audio in a mp3 file named
    #myobj.save("training_"+str(file_index)+".mp3")
    file1 = open("dev_words.txt", "a")
    file1.write(event_generated+'\n')
    file2 = open("dev_frame.txt", "a")
    file2.write(frame+'\n')
    file3 = open("dev.txt", "a")
    file3.write(event_frame+'\n')
    file4 = open("dev_bio.txt", "a")
    file4.write(event_bio+'\n')
    file5 = open("dev_data_act.txt", "a")
    file5.write(data_act+'\n')
    file6 = open("dev_act.txt", "a")
    file6.write(act+'\n')
    print("training data generated, index: ", file_index)
    file_index += 1
    
file1.close()
file2.close()
file3.close()
file4.close()
file5.close()
file6.close()
  
### Playing the converted file
##os.system("mpg321 welcome.mp3")