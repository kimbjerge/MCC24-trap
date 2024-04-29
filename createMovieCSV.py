# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 21:52:04 2019

Python script to create movies, plot observations 
and create empty background label txt files

@author: Kim Bjerge (Made from scratch)
"""

import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Label names for plot
labelNames = ["Mariehone", "Honningbi", "Stenhumle", "Jordhumle", "Svirrehvid", "Svirregul"]

# Functions to get seconds, minutes and hours from time in predictions
def getSeconds(recTime):
    return int(recTime%100)

def getMinutes(recTime):
    minSec = recTime%10000
    return int(minSec/100)

def getHours(recTime):
    return int(recTime/10000)

# Functions to get day, month and year from date in predictions
def getDay(recDate):
    return int(recDate%100)

def getMonthDay(recDate):
    return int(recDate%10000)

def getMonth(recDate):
    return int(getMonthDay(recDate)/100)

def getYear(recDate):
    return int(recDate/10000)


# Substract filterTime in minutes from recTime, do not handle 00:00:00
def substractMinutes(recTime, filterTime):
    
    minute = getMinutes(recTime)
    
    newRecTime = recTime - int(filterTime)*100
    if minute < filterTime: # No space to substract filterTime
        newRecTime = newRecTime - 4000 # Adjust minutes
    
    return newRecTime

# Filter predictions - if the positions are very close and of same class
# Checked within filterTime in minutes (must be a natural number 0,1,2..60)
# It is is assumed that the new prediction belong to the same object
def filter_prediction(lastPredictions, newPrediction, filterTime):
    
    newObject = True
    
    # Filter disabled
    if filterTime == 0:
        return lastPredictions, newObject
    
    # Update last predictions within filterTime window
    timeWindow = substractMinutes(newPrediction['time'], filterTime)
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
    
    # Append new prediction to last preditions
    newLastPredictions.append(newPrediction)
    
    return newLastPredictions, newObject
    
# Load prediction CSV file
# filterTime specifies in minutes how long time window used
# to decide if predictions belongs to the same object
# probability threshold for each class, default above 50%
def load_predictions(filename, selection = 'All', filterTime=0, threshold=[50,50,50,50,50,50]):
    
    file = open(filename, 'r')
    content = file.read()
    file.close()
    splitted = content.split('\n')
    lines = len(splitted)
    foundObjects = []
    lastObjects = []
    for line in range(lines):
        subsplit = splitted[line].split(',')
        if len(subsplit) == 16: # required 15 or 16 data values
            imgname = subsplit[10]
            imgpath = imgname.split('/')
            prob = subsplit[13] # 4
            objClass = int(subsplit[12]) # 5
            # Check selection 
            if (selection == imgpath[0] or selection == 'All'): # and prob >= threshold[objClass-1]:
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
                
                record = {
                        'system': subsplit[0], # 1-5
                        'camera': subsplit[1], # 0 or 1
                        'date' : subsplit[2],
                        'time' : subsplit[3],
                        'prob' : prob, # Class probability 0-100%
                        'class' : objClass, # Classes 1-16
                        'className' : subsplit[11],
                        'valid' : subsplit[14] == 'True',
                        #'id' : int(subsplit[15]), # Index to predictions
                        # Box position and size
                        'confSpecies' : prob, 
                        'classSpecies' : objClass,
                        'speciesName' : subsplit[11],
                        'x1' : x1,
                        'y1' : y1,
                        'x2' : x2,
                        'y2' : y2,
                        'xc' : xc,
                        'yc' : yc,
                        'w' : width,
                        'h' : height,
                        'path' : imgname,
                        'image' : imgpath[3],
                        'label' : 0} # Class label (Unknown = 0)
                
                lastObjects, newObject =  filter_prediction(lastObjects, record, filterTime)
                if newObject:
                    foundObjects.append(record)
            
    return foundObjects

# Load prediction CSV file with header, oder and moth species classifications
# filterTime specifies in minutes how long time window used
# to decide if predictions belongs to the same object
# probability threshold for each class, default above 50%
def load_species_predictions(filename, selection = 'All', filterTime=0, threshold=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], scoresFilename=None):
    
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
                
                record = {
                        'system': subsplit[0], # 1-5
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
                                            
                lastObjects, newObject = filter_prediction(lastObjects, record, filterTime)
                if newObject:
                    foundObjects.append(record)                      
            
    return foundObjects    

    
def countPredictClasses(predictions):
    countClasses = []
    for i in range(len(labelNames)):
        countClasses.append(0);
        
    for object in predictions:
        countClasses[object['class']-1] += 1
    
    return countClasses    
    
def findObjectsInImage(filename, predictions):
    
    found = False
    objectsFound = []
    for object in predictions:
        if object['image'] == filename:
            objectsFound.append(object)
            found = True
    
    return found, objectsFound

def write_to_movie(movie_writer, image_dic, predictions, dim):
            
    for filename in sorted(os.listdir(image_dic)):
        if filename.endswith('.jpg'):
            found, objects = findObjectsInImage(filename, predictions)
            if found:
                print(filename)
                img = cv2.imread(image_dic+filename)
                for insect in objects:
                    
                    confidence = insect['prob']
                    classNameSplit = insect['className'].split(' ')
                    if len(classNameSplit) == 2:
                        className = classNameSplit[1]
                    else:
                        className = insect['className']
                    
                    color = (0,255,255) # Yellow
                    if className == 'Macros':
                        color = (0,255,0) # Green
                        if 'Macros' not in insect['speciesName']:
                            className = insect['speciesName']
                            confidence = insect['confSpecies']
                    if className == 'Micros':
                        color = (255,0,0) # Blue
                        if 'Micros' not in insect['speciesName']:
                            className = insect['speciesName']
                            confidence = insect['confSpecies']
                        
                    if insect['valid'] == False:
                        color = (0,0,255) # Red
                    
                    cv2.rectangle(img,(insect['x1'],insect['y1']-10),(insect['x2'],insect['y2']), color, 4)
                    insectName = className + ' (' + str(confidence)+ ')'
                    y = int(round(insect['y1']-20))
                    cv2.putText(img, insectName, (insect['x1'],y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

                    cv2.putText(img, insect['image'], (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                    #dateTimeStr =  insect['date'] + ' ' + insect['time']
                    #cv2.putText(img, dateTimeStr, (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                
                #cv2.imshow('image',img)
                #v2.waitKey(0)
                #cv2.destroyAllWindows()
                image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                movie_writer.write(image)
            
def create_movie(movie_name, image_dic, predictions, fps=2, size = (3840,2160), scale = 0.5):

    dim = (int(size[0]*scale), int(size[1]*scale))
    movie_writer = cv2.VideoWriter(movie_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, dim)
    write_to_movie(movie_writer, image_dic, predictions, dim)
    movie_writer.release()

def plot_detected_boxes(img, objects, downsize):

    img = cv2.resize(img, (0,0), fx=1/downsize, fy=1/downsize) #, cv2.INTER_CUBIC)
    for insect in objects:
        x1 = int(round(insect['x1']/downsize))
        y1 = int(round((insect['y1']-10)/downsize))
        x2 = int(round(insect['x2']/downsize))
        y2 = int(round(insect['y2']/downsize))
        cv2.rectangle(img,(x1,y1),(x2,y2), (0,0,255), 4)
        insectName = labelNames[insect['class']-1] + '(' + str(insect['prob'])+ ')'
        y = int(round((insect['y1']-20)/downsize))
        cv2.putText(img, insectName, (x1,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    return img

# Show predictions found in predictions list
# Use key=s to save empty lable txt file for background image with wrong predictions       
def show_predictions(image_dic, predictions, downsize=1.5):
    
    count = 0
    for filename in os.listdir(image_dic):
        if filename.endswith('.jpg'):
            found, objects = findObjectsInImage(filename, predictions)
            if found:
                img = cv2.imread(image_dic+'/'+filename)
                img = plot_detected_boxes(img, objects, downsize)
                cv2.imshow('image',img)
                #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                #cv2.resizeWindow('image', w, h)
                #cv2.waitKey(0)
                key = cv2.waitKey(0)
                if key == ord('s'):
                    name = filename.split('.')
                    txtfilename = image_dic+'/'+name[0]+'.txt'
                    print('Create lable file:', txtfilename)
                    open(txtfilename, 'a').close()
                    count = count + 1
                if key == ord('e'):
                    print('Exit')
                    cv2.destroyAllWindows()
                    break
                cv2.destroyAllWindows()
    
    return count

# Show predictions found in predictions list where also lable txt file exist
# Use key=s to delete lable txt file for if wrong background image        
def show_labled_predictions(image_dic, predictions, downsize=1.5):
    
    count = 0
    for filename in os.listdir(image_dic):
        if filename.endswith('.jpg'):
            name = filename.split('.')
            txtfilename = image_dic+'/'+name[0]+'.txt'
            found, objects = findObjectsInImage(filename, predictions)
            if found and os.path.isfile(txtfilename):
                img = cv2.imread(image_dic+'/'+filename)
                img = plot_detected_boxes(img, objects, downsize)
                cv2.imshow('image',img)
                key = cv2.waitKey(0)
                if key == ord('s'):
                    print('Delete lable file:', txtfilename)
                    os.remove(txtfilename)
                else:
                    count = count + 1
                if key == ord('e'):
                    print('Exit')
                    cv2.destroyAllWindows()
                    break
                cv2.destroyAllWindows()
                
    return count

def show_xy_histogram3(predictions, area_name, width = 4224, height = 2376):
    
    # Create data
    g1xc = []
    g1yc = []
    g2xc = []
    g2yc = []
    g3xc = []
    g3yc = []
    g4xc = []
    g4yc = []
    for obj in predictions:
        if obj['class'] == 1:
            g1xc.append(obj['xc'])
            g1yc.append(obj['yc'])
        if obj['class'] == 2: # Honningbi
            g2xc.append(obj['xc'])
            g2yc.append(obj['yc'])
        if obj['class'] == 3 or obj['class'] == 4: # Havehumle og sten/jordhumle
            g3xc.append(obj['xc'])
            g3yc.append(obj['yc'])
        if obj['class'] == 5 or obj['class'] == 6: # Svirrefluer hvid+gul
            g4xc.append(obj['xc'])
            g4yc.append(obj['yc'])
    
    dataxc = (g1xc, g2xc, g3xc, g4xc)
    datayc = (g1yc, g2yc, g3yc, g4yc)
    colors = ("red", "green", "blue", "grey")
    groups = ("Mariehøne", "Honningbi", "Humlebi", "Svirreflue")
    
    # Create plot
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
    
    for dataxc, datayc, color, group in zip(dataxc, datayc, colors, groups):
        ax.scatter(dataxc, datayc, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
    
    plt.title('Scatter plot of insect locations in: ' + area_name)
    plt.legend(loc=2)
    plt.show()
    fig.savefig(area_name)   
    plt.close(fig)   

# Creates a scatter plot of positions of insects found in images
def areaScatterPlots(path, system_name, csv_filename, camera, filterTime=15, show_scatter=False, make_movie=False, threshold=[50,50,50,50,50,50]):
    
    directory = path + system_name + '/'
    
    predictions = load_predictions(csv_filename, filterTime=filterTime, threshold=threshold) 

    if show_scatter:
        area_name = 'areas/' + system_name + ' '  + camera  + '.jpg'
        show_xy_histogram3(predictions, area_name)
   
    if make_movie:
        movie_name = 'movies/' + system_name + ' ' + camera + '.avi'
        image_dic = directory + camera + '/'
        print("Creating movie as:", movie_name)
        create_movie(movie_name, image_dic, predictions) # Create movie with bounding box
        
    print("System:", system_name + "/" + camera)
    print(labelNames)
    print("-------------------------------------------------------------------------------")
    countTotal = []
    for i in range(len(labelNames)):
        countTotal.append(0);
    i = 0
    for predict in predictions: 
        countTotal[predict['class']-1] += 1
        i += 1
    print(countTotal)
    print("==============================")
                
    return countTotal

# Creates a movie with images of bounding boxes for insects found in directory "dateCamera"
def createMovie(path, system_name, model_name, video_path, dateCamera, filterTime=15, threshold=[50,50,50,50,50,50]):

    directory = path + system_name + '/'
    result_file = directory + system_name +'-' + model_name + '.csv'

    movie_name = video_path + system_name + '-'  + dateCamera + '-' + model_name + '.avi'
    predictions = load_predictions(result_file, selection = dateCamera, filterTime = filterTime, threshold=threshold) 
    show_xy_histogram3(predictions, movie_name)
    image_dic = directory + dateCamera + '/'
    create_movie(movie_name, image_dic, predictions) # Create movie with bounding boxes

# Used to show predictions in images with possibility to create background images for training
def showPredictions(path, system_name, model_name, dateCamera):

    directory = path + system_name + '/'
    result_file = directory + system_name +'-' + model_name + '.csv'

    image_dic = directory + dateCamera + '/'
    predictions = load_predictions(result_file, selection = dateCamera) 
    count = show_predictions(image_dic, predictions) # Show images with bounding boxes, possible to save 's' empty lable txt file
    #count = show_labled_predictions(image_dic, predictions) # Show images with bounding boxes, where label txt file exist, possible to delete 's' wrong background image 
    print('Counted labels:', count);

# Plots number of bees and svirrefluer as function of dates where insects found
def plotInsectsDate(path, system_name, model_name, camera, filterTime=15, threshold=[50,50,50,50,50,50]):
    
    directory = path + system_name + '/'
    result_file = directory + system_name +'-' + model_name + '.csv'
    predictions = load_predictions(result_file, selection = 'All', filterTime = filterTime, threshold=threshold) 
    
    currDate = 0
    monthArray = []
    marie = []
    bees = []
    humle = []
    svirre = []
    dayArray = []
    idx = -1
    for predict in predictions:
        if camera == predict['camera']:
            if currDate != predict['date']:
                currDate = predict['date']
                monthArray.append(getMonthDay(currDate))
                marie.append(0)
                bees.append(0)
                humle.appen(0)
                svirre.append(0)
                idx += 1
                dayArray.append(idx)
            classObj = predict['class']
            if classObj == 1: #mariehøne (1)
                marie[idx] += 1
            if classObj == 2: #honnigbi (2)
                bees[idx] += 1
            if classObj >= 3 and classObj <= 4: #stenhumle (3), jordhumle (4)
                humle[idx] += 1
            if classObj >= 5 and classObj <= 6: #svirrehvid (5), svirregul (6)
                svirre[idx] += 1
  
    fig = plt.figure(figsize=(17,15))
    ax = fig.add_subplot(2, 1, 1, axisbg="1.0")         
    ax.plot(dayArray, marie, 'ro', label='Mariehøne')
    ax.plot(dayArray, bees, 'go', label='Honningbi')
    ax.plot(dayArray, humle, 'bo', label='Humlebi')
    ax.plot(dayArray, svirre, 'yo', label='Svirreflue')
    ax.legend(loc=2)
    ax.set_ylim(0, 500)
    ax.set_xlabel('Dage')
    ax.set_ylabel('Antal')
    ax.set_title('Insekter fra ' + str(system_name) + ' Camera ' + str(camera))
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    
    #fig = plt.figure(figsize=(15,15))
    #ax = fig.add_subplot(2, 1, 1, axisbg="1.0")         
    #ax.plot(dayArray, svirre, 'bo')
    #ax.set_ylim(0, 500)
    #ax.set_xlabel('Dage')
    #ax.set_ylabel('Svirrefluer')
    #ax.set_title('Svirrefluer fra ' + str(system_name) + ' Camera ' + str(camera))
    #ax.grid(True)   
    #fig.tight_layout()
    #plt.show()

    print("Dates:", monthArray) 
    
# Create an array with month and day for whole periode
def createPeriode(periode):
    
    monthDayArray = []
    currDate = periode[0]
    while currDate <= periode[1]+1:
        monthDayArray.append(currDate)
        currDate += 1
        month = getMonth(currDate)
        if getDay(currDate) == 31 and (month == 6 or month == 9): # June and September 30 days
            currDate += (100-30);
        if getDay(currDate) == 32 and (month == 5 or month == 7 or month == 8): # May, July and August 31 days
            currDate += (100-31);
            
    return monthDayArray

# Get index that belongs to date
def getDateIdx(currMonthDay, monthDayArray):
    
    for idx in range(len(monthDayArray)):
        if currMonthDay == monthDayArray[idx]:
            return idx

    return 0

# Fundtion to create format of x-axis
globalMonthDayArray = createPeriode([624, 1030])

@ticker.FuncFormatter
def major_formatter(x, pos):
    day = int(globalMonthDayArray[int(x)] % 100)
    month = int(globalMonthDayArray[int(x)] / 100)
    string =  "{}/{}-2019"
    return string.format(day, month) #"%d" % day

            
# Plots number of bees and svirrefluer as function of periode with all dates
def plotInsectsPeriode(path, system_name, model_name, periode, camera, filterTime=15, threshold=[50,50,50,50,50,50]):
    
    directory = path + system_name + '/'
    result_file = directory + system_name +'-' + model_name + '.csv'
    predictions = load_predictions(result_file, selection = 'All', filterTime = filterTime, threshold=threshold) 
    
    #currDate = 0
    monthArray = createPeriode(periode)
    length = len(monthArray)
    marie = np.zeros((length,), dtype=int)
    bees = np.zeros((length,), dtype=int)
    humle = np.zeros((length,), dtype=int)
    svirre = np.zeros((length,), dtype=int)
    dayArray = range(length)
    idx = -1
    for predict in predictions:
        if camera == predict['camera']:
            #if currDate != predict['date']:
            #    currDate = predict['date']
            #    monthArray.append(getMonthDay(currDate))
            #    bees.append(0)
            #    svirre.append(0)
            #    idx += 1
            #    dayArray.append(idx)
            idx = getDateIdx(getMonthDay(predict['date']), monthArray)
            classObj = predict['class']
            if classObj == 1: #mariehøne (1)
                marie[idx] += 1
            if classObj == 2: #honnigbi (2)
                bees[idx] += 1
            if classObj >= 3 and classObj <= 4: #stenhumle (3), jordhumle (4)
                humle[idx] += 1
            if classObj >= 5 and classObj <= 6: #svirrehvid (5), svirregul (6)
                svirre[idx] += 1
  
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(2, 1, 1, axisbg="1.0")         
    ax.plot(dayArray, marie, 'r', label='Mariehøns')
    ax.plot(dayArray, bees, 'g', label='Honningbier')
    ax.plot(dayArray, humle, 'b', label='Humlebier')
    ax.plot(dayArray, svirre, 'y', label='Svirrefluer')
    ax.legend(loc=2)
    ax.xaxis.set_major_formatter(major_formatter)
    ax.set_ylim(0, 500)
    ax.set_xlabel('Dato')
    ax.set_ylabel('Antal')
    ax.set_title('Insekter fra ' + str(system_name) + ' kamera ' + str(camera))
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    fig.savefig('insects/' + str(system_name) + '_' + str(camera) + '-' + str(model_name) + '.jpg')   
    plt.close(fig)   
    
    return [dayArray, monthArray, marie, bees, humle, svirre]

# Plots the sum of all bees and svirre from cameras in list
def plotAllInsectsPeriode(cameras, model_name):
    
    first = True
    for camera in cameras:
        dayArray = camera[0]
        #monthArray = camera[1]
        marie = camera[2]
        bees = camera[3]
        humle = camera[4]
        svirre = camera[5]
        if first:
            marieTotal = marie
            beesTotal = bees
            humleTotal = humle
            svirreTotal = svirre
            first = False
        else:
            for idx in dayArray:
                marieTotal[idx] += marie[idx]
                beesTotal[idx] += bees[idx]
                humleTotal[idx] += humle[idx]
                svirreTotal[idx] += svirre[idx]
            
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(2, 1, 1, axisbg="1.0")         
    ax.plot(dayArray, marieTotal, 'r', label='Mariehøns')
    ax.plot(dayArray, beesTotal, 'g', label='Honningbier')
    ax.plot(dayArray, humleTotal, 'b', label='Humlebier')
    ax.plot(dayArray, svirreTotal, 'y', label='Svirrefluer')
    ax.legend(loc=2)
    ax.xaxis.set_major_formatter(major_formatter)
    #ax.set_ylim(0, 600)
    ax.set_xlabel('Dato')
    ax.set_ylabel('Antal')
    ax.set_title('Insekter fra 10 kameraer (1,05 m2)')
    ax.grid(True)
    fig.tight_layout()
    plt.show()   
    #print("Dates:", monthArray) 
    fig.savefig("insects/TotalAllCameras-" + model_name + ".jpg")   
    plt.close(fig)      
        
                
if __name__=='__main__': 
    
    #path = 'O:/Tech_TTH-KBE/MAVJF/Annotations/2022/dataset3/'
    #predictions = load_predictions('./Moths/dataset3.csv')
    #create_movie('./movies/dataset3.avi', path, predictions)
   
    # Order classifier
    #path = 'O:/Tech_TTH-KBE/MAVJF/data/2022/OH3/20220723/'
    #predictions = load_predictions('./CSV/M2022/20220723.csv')
    #create_movie('./movies/OH3_20220723.avi', path, predictions)

    # Order and species classifier
    path = 'O:/Tech_TTH-KBE/MAVJF/data/2022/snapLV2/'
    predictions = load_species_predictions('./CSV/M2022S/snapLV2.csv')
    create_movie('./movies/snapLV2.avi', path, predictions)
    
    #totalPredictions50 = areaScatterPlots(path, system_name, '0510-22', filterTime=15, show_scatter=False, make_movie=False)
    # predictions = "./Moths"
    # for csv_file in sorted(os.listdir(predictions)):
    #     dirnames = csv_file.split(' ')
    #     system_name = dirnames[0]
    #     camera_name = dirnames[1].split('.')[0]
    #     csv_filename = predictions + csv_file
    #     totalPredictionsVar = areaScatterPlots(path, system_name, csv_filename, camera_name, filterTime=15, show_scatter=True, make_movie=True, threshold=threshold)
  
    # #print("Total YOLO3 model 0510-22:", totalPredictions50, " fixed threshold: 50%")    
    # print("Total YOLO3 model 0510-22:", totalPredictionsVar, " variable threshold (%):", threshold)    
    
    # Predicted and trained models for all camera systems
    """
    model_name = '0510-22' # Best model 50%
    periode = [624, 930] # From 24. june to 30. september
    cameras = []
    cameras.append(plotInsectsPeriode(path, 'S1', model_name, periode, camera=0, filterTime=15,  threshold=threshold)) # Camera 0
    cameras.append(plotInsectsPeriode(path, 'S2', model_name, periode, camera=0, filterTime=15,  threshold=threshold)) # Camera 0
    plotAllInsectsPeriode(cameras, model_name)
    """
    
    #createMovie(path, system_name, model_name, video_path='videos/', dateCamera = 'Juli15_0', filterTime=15, threshold=threshold)
    #showPredictions(path, system_name, model_name, dateCamera = 'Sep06_0') # s-save, e-exit

    
 