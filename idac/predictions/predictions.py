# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 12:48:13 2020

@author: Kim Bjerge
"""

import math
import numpy as np
from idac.objectOfInterrest import ObjectOfInterrest

class Predictions:
    
    def __init__(self, conf):
        print('predictions')
        self.config = conf["classifier"]
        self.species = self.config["species"]
        self.noPredicions = 0
        self.noFilteredPredicions = 0


    def getPredictions(self):
        return self.noPredicions, self.noFilteredPredicions
        
    # Functions to get seconds, minutes and hours from time in predictions
    def getSeconds(self, recTime):
        return int(recTime%100)
    
    def getMinutes(self, recTime):
        minSec = recTime%10000
        return int(minSec/100)
    
    def getHours(self, recTime):
        return int(recTime/10000)
    
    def getTimesec(self, recTime):
        
        timestamp = self.getSeconds(recTime)
        timestamp += self.getMinutes(recTime)*60
        timestamp += self.getHours(recTime)*3600
        return timestamp
    
    # Functions to get day, month and year from date in predictions
    def getDay(self, recDate):
        return int(recDate%100)
    
    def getMonthDay(self, recDate):
        return int(recDate%10000)
    
    def getMonth(self, recDate):
        return int(self.getMonthDay(recDate)/100)
    
    def getYear(self, recDate):
        return int(recDate/10000)
        
    # Substract filterTime in minutes from recTime, do not handle 00:00:00
    def substractMinutes(self, recTime, filterTime):
        
        minute = self.getMinutes(recTime)
        
        newRecTime = recTime - int(filterTime)*100
        if minute < filterTime: # No space to substract filterTime
            newRecTime = newRecTime - 4000 # Adjust minutes
        
        return newRecTime
    
    def addTimes(self, recTime1, recTime2):
        
        seconds = self.getSeconds(recTime1) + self.getSeconds(recTime2) 
        minutesSec = 0 
        if seconds > 59:
            minutesSec = 1
            seconds = seconds%60
            
        minutes  = self.getMinutes(recTime1) + self.getMinutes(recTime2) + minutesSec
        hoursMin = 0 
        if minutes > 59:
            hoursMin = 1
            minutes = minutes%60
            
        hours  = self.getHours(recTime1) + self.getHours(recTime2) + hoursMin
        if hours > 23:
            hours = 0
        
        recTime = hours*10000 + minutes*100 + seconds       
        return recTime
             
    # Filter predictions - if the positions are very close and of same class
    # Checked within filterTime in minutes (must be a natural number 0,1,2..60)
    # It is is assumed that the new prediction belong to the same object
    def filter_prediction(self, lastPredictions, newPrediction, filterTime):
        
        newObject = True
        
        # Filter disabled
        if filterTime == 0:
            self.noPredicions += 1
            return lastPredictions, newObject
        
        # Update last predictions within filterTime window
        timeWindow = self.substractMinutes(newPrediction['time'], filterTime)
        newLastPredictions = []
        for lastPredict in lastPredictions:
            # Check if newPredition is the same date as last predictions and within time window
            if (lastPredict['date'] == newPrediction['date']) and (lastPredict['time'] > timeWindow):
                newLastPredictions.append(lastPredict)
        
        # Check if new predition is found in last Preditions - nearly same position and class
        for lastPredict in newLastPredictions:
            # Check if new prediction is of same class
            if lastPredict['class'] == newPrediction['class']:
                xlen = lastPredict['xc'] - newPrediction['xc']
                ylen = lastPredict['yc'] - newPrediction['yc']
                # Compute distance between predictions
                dist = math.sqrt(xlen*xlen + ylen*ylen)
                #print(dist)
                if dist < 25: # NB adjusted for no object movement
                    # Distance between center of boxes are very close
                    # Then we assume it is not a new object
                    newObject = False
        
        self.noPredicions += 1
        if newObject == False:
            self.noFilteredPredicions += 1

        # Append new prediction to last preditions
        newLastPredictions.append(newPrediction)
        
        return newLastPredictions, newObject
        
    # Load prediction CSV file
    # filterTime specifies in minutes how long time window used
    # to decide if predictions belongs to the same object
    # probability threshold for each class, default above 50%
    def load_predictions(self, filename, selection = 'All', filterTime=0, threshold=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], scoresFilename=None):
        
        if scoresFilename is None:
            predictScores = None
        else:
            predictScores = np.load(scoresFilename, allow_pickle=True)
     
        file = open(filename, 'r')
        content = file.read()
        file.close()
        splitted = content.split('\n')
        lines = len(splitted)
        foundObjects = []
        lastObjects = []
        for line in range(lines):
            subsplit = splitted[line].split(',')
            if len(subsplit) == 16: # required 11 or 16 data values
                imgname = subsplit[10]
                imgpath = imgname.split('/')
                prob = float(subsplit[13]) # 4
                objClass = int(subsplit[12])+1 # 5
                #prob = int(subsplit[4])
                #objClass = int(subsplit[5])
                # Check selection 
                if (selection == imgpath[0] or selection == 'All') and prob >= threshold[objClass-1]:
                    x1 = int(subsplit[6])
                    y1 = int(subsplit[7])
                    x2 = int(subsplit[8])
                    y2 = int(subsplit[9])
                    # Convert points of box to YOLO format: center point and w/h
                    width = x2-x1
                    height = y2-y1
                    xc = x1 - round(width/2)
                    if xc < 0: xc = 0
                    yc = y1 - round(height/2)
                    if yc < 0: yc = 0
                    key = int(subsplit[15]) # Index to prediction scores
                    if predictScores is None:
                        scores = None
                    else:
                        scores = predictScores[key]
                    
                    record = {'system': subsplit[0], # 1-5
                            'camera': subsplit[1], # 0 or 1
                            'date' : int(subsplit[2]),
                            'time' : int(subsplit[3]),
                            'timeRec' : int(subsplit[3]),
                            'prob' : prob, # Class probability 0-100%
                            'class' : objClass, # Classes 0-15
                            'className' : subsplit[11],
                            'valid' : subsplit[14] == 'True',
                            'key' : key, # Index to predictions
                            'scores' : scores,
                            # Box position and size
                            'x1' : x1,
                            'y1' : y1,
                            'x2' : x2,
                            'y2' : y2,
                            'xc' : xc,
                            'yc' : yc,
                            'w' : width,
                            'h' : height,
                            'image' : imgpath[3],
                            'pathimage' : subsplit[10],
                            'label' : 0} # Class label (Unknown = 0)
                            
                    
                    lastObjects, newObject =  self.filter_prediction(lastObjects, record, filterTime)
                    if newObject:
                        foundObjects.append(record)                      
                
        return foundObjects    

    # Load prediction CSV file with header, oder and moth species classifications
    # filterTime specifies in minutes how long time window used
    # to decide if predictions belongs to the same object
    # probability threshold for each class, default above 50%
    def load_species_predictions(self, filename, selection = 'All', filterTime=0, threshold=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], scoresFilename=None):
        
        if scoresFilename is None:
            predictScores = None
        else:
            predictScores = np.load(scoresFilename, allow_pickle=True)
     
        file = open(filename, 'r')
        content = file.read()
        file.close()
        splitted = content.split('\n')
        lines = len(splitted)
        foundObjects = []
        lastObjects = []
        first=True
        for line in range(lines):
            subsplit = splitted[line].split(',')
            if first:
                if len(subsplit) != 19:
                    print("Wrong header in CSV file", line)
                first = False
                continue # Skip first line 
            if len(subsplit) == 19: # required 19 data values
                imgname = subsplit[10]
                imgpath = imgname.split('/')
                orderClassName = subsplit[11]
                prob = float(subsplit[13]) # 4
                objClass = int(subsplit[12])+1 # 5
                speciesName = subsplit[16] # Moth species for Lepidoptera Macros and Micros
                classSpecies = int(subsplit[17])
                confSpecies = float(subsplit[18]) # Confidence of species classifier

                #prob = int(subsplit[4])
                #objClass = int(subsplit[5])
                # Check selection 
                if (selection == imgpath[0] or selection == 'All') and prob >= threshold[objClass-1]:
                    x1 = int(subsplit[6])
                    y1 = int(subsplit[7])
                    x2 = int(subsplit[8])
                    y2 = int(subsplit[9])
                    # Convert points of box to YOLO format: center point and w/h
                    width = x2-x1
                    height = y2-y1
                    xc = x1 - round(width/2)
                    if xc < 0: xc = 0
                    yc = y1 - round(height/2)
                    if yc < 0: yc = 0
                    key = int(subsplit[15]) # Index to prediction scores
                    if predictScores is None:
                        scores = None
                    else:
                        scores = predictScores[key]
                    
                    record = {'system': subsplit[0], # 1-5
                            'camera': subsplit[1], # 0 or 1
                            'date' : int(subsplit[2]),
                            'time' : int(subsplit[3]),
                            'timeRec' : int(subsplit[3]),
                            # Oder classification
                            'prob' : prob, # Class probability 0-100%
                            'class' : objClass, # Order classes 0-15
                            'className' : orderClassName,
                            'valid' : subsplit[14] == 'True',
                            'key' : key, # Index to predictions
                            'scores' : scores,
                            # Species classification
                            'confSpecies' : confSpecies, # Confidence score of species classification
                            'classSpecies' : classSpecies, # Moth species ID classified
                            'speciesName' : speciesName, # Moth species name
                            # Box position and size
                            'x1' : x1,
                            'y1' : y1,
                            'x2' : x2,
                            'y2' : y2,
                            'xc' : xc,
                            'yc' : yc,
                            'w' : width,
                            'h' : height,
                            'image' : imgpath[3],
                            'pathimage' : subsplit[10],
                            'label' : 0} # Class label (Unknown = 0)
                                                
                    lastObjects, newObject =  self.filter_prediction(lastObjects, record, filterTime)
                    if newObject:
                        foundObjects.append(record)                      
                
        return foundObjects    
    
    def getValidTimeStamps(self, deltaTime, startTime=230000, endTime=30000):
        
        validTimeStamps = []
        nextTime = startTime
        while  nextTime < endTime or nextTime >= startTime:
            validTimeStamps.append(nextTime)
            nextTime = self.addTimes(nextTime, deltaTime)
            
        return validTimeStamps
    
    def findPredictionsCloseToTimeForward(self, predictions, timeStamp): #V1 forward in time
        
        currentTime = 999999
        foundPredictions = []
        for prediction in predictions:
            if timeStamp < 120000 and prediction['time'] > 120000: # Search for timestamp after midtnight
                continue
            if prediction['time'] >= timeStamp:
                if currentTime == 999999:
                    currentTime = prediction['time']
                elif prediction['time'] != currentTime:
                    break  
                foundPredictions.append(prediction.copy())
                
        return foundPredictions
                 
    def findPredictionsCloseToTimeBackward(self, predictions, timeStamp):
        
        currentTime = 999999
        foundPredictions = []
        if timeStamp < 120000:
            timeStamp += 240000 # Use time from 120000 - 360000 to handle midt night crossing
        for prediction in predictions:
            predictionTime =  prediction['time']
            if predictionTime < 120000:
                predictionTime += 240000 
            if predictionTime <= timeStamp:
                if currentTime == 999999:
                    currentTime = prediction['time']
                elif prediction['time'] != currentTime: 
                    break # Use only detections close to timeStamp
                foundPredictions.append(prediction.copy())
                
        return foundPredictions
    
    def select_predictions(self, predictions, deltaTime=0, startTime=230000, endTime=30000, backward=True):
        
        selectedPredictions=[]
        validTimeStamps = self.getValidTimeStamps(deltaTime, startTime=startTime, endTime=endTime)
        #print(validTimeStamps)

        reverse_predictions= sorted(predictions, key=lambda d: (d['date']*1000000+d['time']), reverse=True)
        
        for timeStamp in validTimeStamps:
            
            if backward:
                selectPredictions = self.findPredictionsCloseToTimeBackward(reverse_predictions, timeStamp)
            else:
                selectPredictions = self.findPredictionsCloseToTimeForward(predictions, timeStamp)
            
            for insect in selectPredictions:
                insect['time'] = timeStamp
                #if timeStamp > 235900 or timeStamp < 200:
                #print(insect['time'], insect['timeRec'], insect['key'])
                selectedPredictions.append(insect)
        
        return selectedPredictions

    # Find bounding boxes and classes found in image by filename
    def findboxes(self, filename, predictions, useSpeciesPredictions=False):
         
        ooi = []
        count = 0
        for predict in predictions:
            if filename == predict['image']:
                obj = ObjectOfInterrest(predict['x1'], predict['y1'], predict['w'], predict['h'])
                obj.confidenceAvg = predict['prob'] # Average confidence
                obj.percent = predict['prob'] 
                #obj.label = predict['className']
                obj.label = self.species[predict['class']-1]
                obj.order = obj.label
                if useSpeciesPredictions:
                    if "Lepidoptera" in obj.label: 
                        # If order of lepidoptera then use species classification name and confidence
                        obj.label = predict['speciesName'].replace('Â','') # Wrong chacter in label???
                        obj.percent = predict['confSpecies']
                obj.features = predict['scores']
                obj.valid = predict['valid']
                obj.key = predict['key']
                obj.timesec = self.getTimesec(predict['time']) 
                #print(obj.label, obj.percent, obj.timesec)
                ooi.append(obj)
                count = count + 1

        return count, ooi

         