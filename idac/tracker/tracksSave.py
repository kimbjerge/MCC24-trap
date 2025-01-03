#import numpy as np
#import math
#from datetime import datetime
#from datetime import timedelta

class TracksSave:
    
    def __init__(self, fileName):
        self.tracksFileName = fileName
        self.tracksFile = open(fileName, 'w')
        headerline = "id,key,date,time,confidence,valid,order,species,xc,yc,x1,y1,width,height,image\n"
        self.tracksFile.write(headerline) 
        
    def cleanText(self, text_with_special_chars):
         
        # Initialize an empty string to store the cleaned text
        cleaned_text = ""
        # Iterate through each character in the input string
        for char in text_with_special_chars:
            # Check if the character is alphanumeric (letters or digits)
            if char.isalnum() or char == ' ' or char == '.' or char == '&' or char == '-':
                # If alphanumeric or a space, add it to the cleaned text
                cleaned_text += char
            else:
                #print("-------------------------------->Error char TracksSave", char)
                cleaned_text += ' ' # Convert special chars to space
                
        return cleaned_text
     
    def save(self, predict, ois):
        for oi in ois:
            species = "none"
            if "Lepidoptera" in oi.order: # Moth
                species = self.cleanText(oi.label)
            line = str(oi.id) + ',' + \
                   str(oi.key) + ',' + \
                   str(predict['date']) + ',' + \
                   str(predict['time']) + ',' + \
                   str(oi.percent) + ',' + \
                   str(oi.valid) + ',' + \
                   oi.order + ',' + \
                   species  + ',' + \
                   str(int(oi.x + oi.w / 2)) + ',' + \
                   str(int(oi.y + oi.h / 2)) + ',' + \
                   str(oi.x) + ',' + \
                   str(oi.y) + ',' + \
                   str(oi.w) + ',' + \
                   str(oi.h) + ',' + \
                   predict['image'] + '\n'
            self.tracksFile.write(line) 
    
    def close(self):
        self.tracksFile.close()
        

