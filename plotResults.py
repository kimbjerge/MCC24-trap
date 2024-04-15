# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:50:12 2024

@author: Kim Bjerge
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import json
import datetime
from scipy.stats import expon
from idac.configreader.configreader import readconfig
from idac.predictions.predictions import Predictions
from scipy.stats import pearsonr

labelNames1 = ["Araneae", "Coleoptera", "Diptera Brachycera", "Diptera Nematocera", "Diptera Tipulidae", 
              "Diptera Trichocera", "Ephemeroptera", "Hemiptera", "Hymenoptera Other", "Hymenoptera Vespidae", 
              "Lepidoptera Macros", "Lepidoptera Micros", "Neuroptera", "Opiliones", "Trichoptera", "Vegetation"]

labelNamesPlot = ["Araneae", "Coleoptera", "Diptera Brachycera", "Diptera Nematocera", "Diptera Tipulidae", 
                  "Diptera Trichocera", "Ephemeroptera", "Hemiptera", "Hymenoptera Other", "Hymenoptera Vespidae", 
                  "Lepidoptera Macros", "Lepidoptera Micros", "Neuroptera", "Opiliones", "Trichoptera"]

#labelNames = ["Diptera Nematocera", "Lepidoptera Macros", "Lepidoptera Micros", "Trichoptera"]
labelNames = ["Lepidoptera Macros", "Lepidoptera Micros"]

config_filename = './ITC_config.json'
trackPath = "./tracks/"
csvPath = "./CSV/M2022/"

class timedate:
    
    def __init__(self):
        print('timeDate')
        
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
    
    def strMonthDay(self, recDate):
        text = str(self.getDay(recDate)) + '/' + str(self.getMonth(recDate))
        return text
    
    def getDayOfYear(self, recDate):
        date = datetime.datetime(self.getYear(recDate), self.getMonth(recDate), self.getDay(recDate))
        return date.strftime('%j')
        
    
def createDatelist(dataset):

    td = timedate()
    currentDate = 0
    nextDate = 0
    dates = []
    dayOfYears = []
    for i, obj in dataset.iterrows():
        if currentDate != obj['startdate']:
            if nextDate != obj['startdate']:
                print("NextDate", nextDate)
            currentDate = obj['startdate']
            nextDate = currentDate + 1
            dates.append(currentDate)
            dayOfYear = td.getDayOfYear(currentDate)
            #print(dayOfYear)
            dayOfYears.append(int(dayOfYear))

    return dates, dayOfYears

    
def countAbundance(dataset, dates):

    abundance = np.zeros(len(dates)).tolist()
    for i, obj in dataset.iterrows():
        dateIdx = dates.index(obj['startdate'])
        abundance[dateIdx] += 1

    return abundance
  
def countSnapAbundance(predicted, dates, labelName, valid=True):

    abundance = np.zeros(len(dates)).tolist()
    for insect in predicted:
        if insect["className"] == labelName:
            if valid == False or insect['valid'] == True:
                if insect['date'] in dates:
                    dateIdx = dates.index(insect['date'])
                    abundance[dateIdx] += 1
                else:
                    print("Date did not exist", insect['date'])

    return abundance    

def getDurationList(dataset, maxLimit):
    
    durations = []
    for i, obj in dataset.iterrows():
        if obj['duration'] < maxLimit:
            durations.append(obj['duration'])
    
    return durations

def loadTrackFiles(trap, countsTh, percentageTh):

    trackFiles = trackPath + trap + '/'  
    dataframes = []
    for fileName in sorted(os.listdir(trackFiles)):
        if "TR.csv" in fileName:
            #print(trap, trackFiles + fileName)
            #data_df = pd.read_json(trackFiles + fileName)
            data_df = pd.read_csv(trackFiles + fileName)
            dataframes.append(data_df)                
    dataset = pd.concat(dataframes)
    print(trap, len(dataset))
    dateList, dayOfYear = createDatelist(dataset)
    selDataset1 = dataset.loc[dataset['percentage'] > percentageTh]
    selDataset2 = selDataset1.loc[selDataset1['counts'] >= countsTh]
    
    return dateList, dayOfYear, selDataset2

def loadSnapFiles(trap):
    
    conf = readconfig(config_filename)
    predict = Predictions(conf)
    trackSnapFile = csvPath + "snap" + trap + ".csv"
    threshold=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    predicted = predict.load_predictions(trackSnapFile, filterTime=0, threshold=threshold)
    
    return predicted

def plotTimeHistograms(traps, trackPath, countsTh, percentageTh, labelNames, resultFileName, color):
    
    td = timedate()
    
    figure = plt.figure(figsize=(12,12))
    figure.tight_layout(pad=1.0)

    idxFig = 1
    for trap in traps:
        
        dateList, dayOfYear, selDataset2 = loadTrackFiles(trap, countsTh, percentageTh)
        
        # trackFiles = trackPath + trap + '/'  
        # dataframes = []
        # for fileName in sorted(os.listdir(trackFiles)):
        #     if "TR.csv" in fileName:
        #         #print(trap, trackFiles + fileName)
        #         #data_df = pd.read_json(trackFiles + fileName)
        #         data_df = pd.read_csv(trackFiles + fileName)
        #         dataframes.append(data_df)                
        # dataset = pd.concat(dataframes)
        # print(trap, len(dataset))
        # dateList, dayOfYear = createDatelist(dataset)
        # selDataset1 = dataset.loc[dataset['percentage'] > percentageTh]
        # selDataset2 = selDataset1.loc[selDataset1['counts'] >= countsTh]
        
        if "ax" in locals():
            ax = figure.add_subplot(2, 2, idxFig, sharex = ax, sharey = ax) 
        else:
            ax = figure.add_subplot(2, 2, idxFig)     
        
        title = trap
        labels = ""
        colorIdx = 0
        #colors = ["green", "blue", "orange", "brown"]
        for labelName in labelNames:
            selDataset = selDataset2.loc[selDataset2['class'] == labelName]
            print(trap, labelName, len(selDataset))
            durations = getDurationList(selDataset, maxLimit=1500)

            labelText = labelName 
            ax.hist(durations, bins=150, label=labelText, color=color)
            labels += ' ' + labelName
            colorIdx += 1  
            
            #https://ipython-books.github.io/75-fitting-a-probability-distribution-to-data-with-the-maximum-likelihood-method/
            meanDuration = np.mean(durations)
            population_rate = 1/meanDuration
            xs = np.arange(0, 1500, 1500/150)
            ys = expon.pdf(xs, scale=1./population_rate)
            zs = ys*100000
            ax.plot(xs, zs, label='Estimate')
            
            # sample_size = 100            
            # get_sample = lambda n: np.random.exponential(population_rate, n)
            # sample = get_sample(sample_size)
            # plt.hist(sample, density=True, label='sample')            
            # plt.legend()
            # plt.show()

      
            
        title += " (" + td.strMonthDay(dateList[0]) + '-' + td.strMonthDay((dateList[-1])) + ")"
        ax.set_title(title)
        ax.set_xlim(0, 1200)
        if idxFig == 1:
            ax.legend()
        
        ax.set_yscale('log')
        if idxFig == 3 or idxFig == 4: 
            ax.set_xlabel('Seconds')
        if idxFig == 1 or idxFig == 3:
            ax.set_ylabel('Frequency')
        
        idxFig += 1        
    
    plt.savefig("./results/" + resultFileName)
    plt.show() 


def plotSnapAbundance(traps, trackPath, csvPath, countsTh, percentageTh, labelNames, resultFileName, colorOffset=0):
    
    td = timedate()
    
    figure = plt.figure(figsize=(12,12))
    figure.tight_layout(pad=1.0)
    # conf = readconfig(config_filename)
    # predict = Predictions(conf)
    
    idxFig = 1
    for trap in traps:
        trackFiles = trackPath + trap + '/'
        
        dateList, dayOfYear, selDataset2 = loadTrackFiles(trap, countsTh, percentageTh)
        
        # dataframes = []
        # for fileName in sorted(os.listdir(trackFiles)):
        #     if "TR.csv" in fileName:
        #         #print(trap, trackFiles + fileName)
        #         #data_df = pd.read_json(trackFiles + fileName)
        #         data_df = pd.read_csv(trackFiles + fileName)
        #         dataframes.append(data_df)
        # dataset = pd.concat(dataframes)
        # print(trap, len(dataset))
        # dateList, dayOfYear = createDatelist(dataset)
        # selDataset1 = dataset.loc[dataset['percentage'] > percentageTh]
        # selDataset2 = selDataset1.loc[selDataset1['counts'] >= countsTh]
        
        if "ax" in locals():
            ax = figure.add_subplot(2, 2, idxFig, sharex = ax, sharey = ax) 
        else:
            ax = figure.add_subplot(2, 2, idxFig) 

        predicted = loadSnapFiles(trap)
        # trackSnapFile = csvPath + "snap" + trap + ".csv"
        # threshold=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        # predicted = predict.load_predictions(trackSnapFile, filterTime=0, threshold=threshold)               

        title = trap
        labels = ""
        colorIdx = 0
        colors = ["green", "blue", "cyan"]
        colorsSnap = ["orange", "red", "brown"]
        for labelName in labelNames:
            selDataset = selDataset2.loc[selDataset2['class'] == labelName]
            print(trap, labelName, len(selDataset))
            abundance = countAbundance(selDataset, dateList)
            abundanceSnap = countSnapAbundance(predicted, dateList, labelName)
            correlation, _ = pearsonr(abundance, abundanceSnap)
            correlation = np.round(correlation * 100)/100
            
            labelText = labelName + ' (track)'
            ax.plot(dayOfYear, abundance, label=labelText, color=colors[colorIdx+colorOffset])
            labelText = labelName + ' (TL)'
            ax.plot(dayOfYear, abundanceSnap, label=labelText, color=colorsSnap[colorIdx+colorOffset])
            labels += ' ' + labelName
            colorIdx += 1
 
        title += "  " + td.strMonthDay(dateList[0]) + '-' + td.strMonthDay((dateList[-1])) + r"  $\rho$=" + str(correlation)
        ax.set_title(title)
        if idxFig == 1:
            ax.legend()
        
        ax.set_yscale('log')
        if idxFig == 3 or idxFig == 4: 
            ax.set_xlabel('Day of Year')
        if idxFig == 1 or idxFig == 3:
            ax.set_ylabel('Observations')
        
        idxFig += 1
    
    plt.savefig("./results/" + resultFileName)
    plt.show() 
    
def plotAbundance(traps, trackPath, countsTh, percentageTh, labelNames, resultFileName):
    
    td = timedate()
    
    figure = plt.figure(figsize=(12,12))
    figure.tight_layout(pad=1.5)

    idxFig = 1
    for trap in traps:
        
        
        dateList, dayOfYear, selDataset2 = loadTrackFiles(trap, countsTh, percentageTh)
        
        # trackFiles = trackPath + trap + '/'
        # dataframes = []
        # for fileName in sorted(os.listdir(trackFiles)):
        #     if "TR.csv" in fileName:
        #         #print(trap, trackFiles + fileName)
        #         #data_df = pd.read_json(trackFiles + fileName)
        #         data_df = pd.read_csv(trackFiles + fileName)
        #         dataframes.append(data_df)           
        # dataset = pd.concat(dataframes)
        # print(trap, len(dataset))
        # dateList, dayOfYear = createDatelist(dataset)
        # selDataset1 = dataset.loc[dataset['percentage'] > percentageTh]
        # selDataset2 = selDataset1.loc[selDataset1['counts'] >= countsTh]
        
        if "ax" in locals():
            ax = figure.add_subplot(2, 2, idxFig, sharex = ax, sharey = ax) 
        else:
            ax = figure.add_subplot(2, 2, idxFig) 
               
        title = trap
        labels = ""
        colorIdx = 0
        colors = ["black", "orange", "orange", "brown"]
        for labelName in labelNames:
            selDataset = selDataset2.loc[selDataset2['class'] == labelName]
            print(trap, labelName, len(selDataset))
            abundance = countAbundance(selDataset, dateList)

            labelText = labelName + ' ' + str(countsTh*2) + 's'
            ax.plot(dayOfYear, abundance, label=labelText, color=colors[colorIdx])
            labels += ' ' + labelName
            colorIdx += 1
 
        title += " (" + td.strMonthDay(dateList[0]) + '-' + td.strMonthDay((dateList[-1])) + ")"
        ax.set_title(title)
        if idxFig == 1:
            ax.legend()
        
        #ax.set_yscale('log')
        if idxFig == 3 or idxFig == 4: 
            ax.set_xlabel('Day of Year')
        if idxFig == 1 or idxFig == 3:
            ax.set_ylabel('Tracks')
        
        idxFig += 1
    
    plt.savefig("./results/" + resultFileName)
    plt.show() 

def plotAbundanceAllClasses(trap, countsTh, percentageTh, resultFileName, useSnapImages=False):

    figure = plt.figure(figsize=(20,20))
    figure.tight_layout(pad=1.0)
    plt.rcParams.update({'font.size': 18})
 
    trackFiles = trackPath + trap + '/'
    
    dateList, dayOfYear, selDataset2 = loadTrackFiles(trap, countsTh, percentageTh)
        
    # dataframes = []
    # for fileName in sorted(os.listdir(trackFiles)):
    #     if "TR.csv" in fileName:
    #         #print(trap, trackFiles + fileName)
    #         #data_df = pd.read_json(trackFiles + fileName)
    #         data_df = pd.read_csv(trackFiles + fileName)
    #         dataframes.append(data_df)        
    # dataset = pd.concat(dataframes)
    # print(trap, len(dataset))
    # dateList, dayOfYear = createDatelist(dataset)
    # selDataset1 = dataset.loc[dataset['percentage'] > percentageTh]
    # selDataset2 = selDataset1.loc[selDataset1['counts'] >= countsTh]
    
    td = timedate()
    subtitle = trap + " (" + td.strMonthDay(dateList[0]) + '-' + td.strMonthDay((dateList[-1])) + ")"
 
    if useSnapImages:    
        predicted = loadSnapFiles(trap)
        
        # conf = readconfig(config_filename)
        # predict = Predictions(conf)
        # trackSnapFile = csvPath + "snap" + trap + ".csv"
        # threshold=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        # predicted = predict.load_predictions(trackSnapFile, filterTime=0, threshold=threshold)
                          
    idxFig = 1
    for labelName in labelNamesPlot:

        if "ax" in locals():
            ax = figure.add_subplot(5, 3, idxFig, sharex = ax) #, sharey = ax) 
        else:
            ax = figure.add_subplot(5, 3, idxFig) 
             
        title = labelName
        colorIdx = 0
        if useSnapImages:
            colors = ["green", "cyan",  "orange", 
                      "orange","green", "cyan", 
                      "green", "cyan",  "orange", 
                      "orange","green", "cyan",  
                      "green", "cyan",  "orange"]
        else:
            colors = ["green", "red", "purple", 
                      "brown", "brown", "purple", 
                      "olive",  "cyan", "orange", 
                      "red",   "green", "blue", 
                      "cyan", "orange", "olive"]
            
                  
        #labelNamesPlot = ["Araneae", "Coleoptera", "Diptera Brachycera", "Diptera Nematocera", "Diptera Tipulidae", 
        #                  "Diptera Trichocera", "Ephemeroptera", "Hemiptera", "Hymenoptera Other", "Hymenoptera Vespidae", 
        #                  "Lepidoptera Macros", "Lepidoptera Micros", "Neuroptera", "Opiliones", "Trichoptera"]
        selDataset = selDataset2.loc[selDataset2['class'] == labelName]
        print(trap, labelName, len(selDataset))
        abundance = countAbundance(selDataset, dateList)

        labelText = labelName #+ ' ' + str(countsTh*2) + 's'
        colorIdx = labelNamesPlot.index(labelName)
        ax.plot(dayOfYear, abundance, label="Tracks", color=colors[colorIdx])
        
        if useSnapImages:    
            abundanceSnap = countSnapAbundance(predicted, dateList, labelName)
            ax.plot(dayOfYear, abundanceSnap, label="TL", color="black")
            correlation, _ = pearsonr(abundance, abundanceSnap)
            correlation = np.round(correlation * 100)/100
            title += r" ($\rho$=" + str(correlation) + ")"
  
        ax.set_title(title)
        if useSnapImages and idxFig == 1:
            ax.legend()  
        if useSnapImages:
            ax.set_yscale('log')
        if idxFig in [13, 14, 15]: 
            ax.set_xlabel('Day of Year')
        ax.set_xlim(180, 320)
        if idxFig in [1, 4, 7, 10, 13]: 
            if useSnapImages:
                ax.set_ylabel('Observations')
            else:
                ax.set_ylabel('Tracks')
        
        idxFig += 1
  
    plt.suptitle(subtitle)
    plt.tight_layout(pad=1.0)
    plt.savefig("./results/" + resultFileName)
    plt.show() 

    
def plotAbundanceSelectedClasses(countsTh, percentageTh):

    labelNames =  ["Lepidoptera Macros", "Lepidoptera Micros"]
    traps = ['OH1', 'OH2', 'OH3', 'OH4']
    plotAbundance(traps, trackPath, countsTh, percentageTh, labelNames, "OH_Lepidoptera.png")
    traps = ['SS1', 'SS2', 'SS3', 'SS4']
    plotAbundance(traps, trackPath, countsTh, percentageTh, labelNames, "SS_Lepidoptera.png")
    traps = ['LV1', 'LV2', 'LV3', 'LV4']
    plotAbundance(traps, trackPath, countsTh, percentageTh, labelNames, "LV_Lepidoptera.png")
    
    labelNames =  ["Diptera Brachycera", "Diptera Nematocera"]
    traps = ['OH1', 'OH2', 'OH3', 'OH4']
    plotAbundance(traps, trackPath, countsTh, percentageTh, labelNames, "OH_Dipetra.png")
    traps = ['SS1', 'SS2', 'SS3', 'SS4']
    plotAbundance(traps, trackPath, countsTh, percentageTh, labelNames, "SS_Dipetra.png")
    traps = ['LV1', 'LV2', 'LV3', 'LV4']
    plotAbundance(traps, trackPath, countsTh, percentageTh, labelNames, "LV_Dipetra.png")

    labelNames =  ["Hymenoptera Other", "Trichoptera"]
    traps = ['OH1', 'OH2', 'OH3', 'OH4']
    plotAbundance(traps, trackPath, countsTh, percentageTh, labelNames, "OH_Other.png")
    traps = ['SS1', 'SS2', 'SS3', 'SS4']
    plotAbundance(traps, trackPath, countsTh, percentageTh, labelNames, "SS_Other.png")
    traps = ['LV1', 'LV2', 'LV3', 'LV4']
    plotAbundance(traps, trackPath, countsTh, percentageTh, labelNames, "LV_Other.png")    
    
def plotTracksVsSnap(countsTh, percentageTh):

    traps = ['SS1', 'SS2', 'SS3', 'SS4']
    labelNames =  ["Lepidoptera Macros"]
    plotSnapAbundance(traps, trackPath,  csvPath, countsTh, percentageTh, labelNames, "SS_LepidopteraMacros_Snap.png", colorOffset=0)
    labelNames =  ["Lepidoptera Micros"]
    plotSnapAbundance(traps, trackPath,  csvPath, countsTh, percentageTh, labelNames, "SS_LepidopteraMicros_Snap.png", colorOffset=1)
    labelNames =  ["Diptera Nematocera"]
    plotSnapAbundance(traps, trackPath,  csvPath, countsTh, percentageTh, labelNames, "SS_DipteraNematocera_Snap.png", colorOffset=2)
    labelNames =  ["Diptera Brachycera"]
    plotSnapAbundance(traps, trackPath,  csvPath, countsTh, percentageTh, labelNames, "SS_DipteraBrachycera_Snap.png", colorOffset=2)
    labelNames =  ["Trichoptera"]
    plotSnapAbundance(traps, trackPath,  csvPath, countsTh, percentageTh, labelNames, "SS_Trichoptera_Snap.png", colorOffset=2)

    traps = ['OH1', 'OH2', 'OH3', 'OH4']
    labelNames =  ["Lepidoptera Macros"]
    plotSnapAbundance(traps, trackPath,  csvPath, countsTh, percentageTh, labelNames, "OH_LepidopteraMacros_Snap.png", colorOffset=0)
    labelNames =  ["Lepidoptera Micros"]
    plotSnapAbundance(traps, trackPath,  csvPath, countsTh, percentageTh, labelNames, "OH_LepidopteraMicros_Snap.png", colorOffset=1)
    labelNames =  ["Diptera Nematocera"]
    plotSnapAbundance(traps, trackPath,  csvPath, countsTh, percentageTh, labelNames, "OH_DipteraNematocera_Snap.png", colorOffset=2)
    labelNames =  ["Diptera Brachycera"]
    plotSnapAbundance(traps, trackPath,  csvPath, countsTh, percentageTh, labelNames, "OH_DipteraBrachycera_Snap.png", colorOffset=2)
    labelNames =  ["Trichoptera"]
    plotSnapAbundance(traps, trackPath,  csvPath, countsTh, percentageTh, labelNames, "OH_Trichoptera_Snap.png", colorOffset=2)
 
    traps = ['LV1', 'LV2', 'LV3', 'LV4']
    labelNames =  ["Lepidoptera Macros"]
    plotSnapAbundance(traps, trackPath,  csvPath, countsTh, percentageTh, labelNames, "LV_LepidopteraMacros_Snap.png", colorOffset=0)
    labelNames =  ["Lepidoptera Micros"]
    plotSnapAbundance(traps, trackPath,  csvPath, countsTh, percentageTh, labelNames, "LV_LepidopteraMicros_Snap.png", colorOffset=1)
    labelNames =  ["Diptera Nematocera"]
    plotSnapAbundance(traps, trackPath,  csvPath, countsTh, percentageTh, labelNames, "LV_DipteraNematocera_Snap.png", colorOffset=2)
    labelNames =  ["Diptera Brachycera"]
    plotSnapAbundance(traps, trackPath,  csvPath, countsTh, percentageTh, labelNames, "LV_DipteraBrachycera_Snap.png", colorOffset=2)
    labelNames =  ["Trichoptera"]
    plotSnapAbundance(traps, trackPath,  csvPath, countsTh, percentageTh, labelNames, "LV_Trichoptera_Snap.png", colorOffset=2)   
    

def plotTimeHistogramsSelectedTrap(traps, trapscountsTh, percentageTh, name):

    traps = ['OH1', 'OH2', 'OH3', 'OH4']
    labelNames =  ["Lepidoptera Macros"]
    plotTimeHistograms(traps, trackPath, 2, 50, labelNames, name + "_LepidopteraMacros_TimeHist.png", "green")
    labelNames =  ["Lepidoptera Micros"]
    plotTimeHistograms(traps, trackPath, 2, 50, labelNames, name + "_LepidopteraMicros_TimeHist.png", "blue")
    labelNames =  ["Diptera Nematocera"]
    plotTimeHistograms(traps, trackPath, 2, 50, labelNames, name + "_DipteraNematocera_TimeHist.png", "orange")
    labelNames =  ["Trichoptera"]
    plotTimeHistograms(traps, trackPath, 2, 50, labelNames, name + "_Trichoptera_TimeHist.png", "brown")
    #labelNames =  ["Hymenoptera Vespidae"]
    #traps = ['SS1', 'SS2', 'SS3', 'SS4']
    plotTimeHistograms(traps, trackPath, 2, 50, labelNames, "SS_Vespidae_TimeHist.png", "orange")


def testSelectPredictions(trap, sampleTime=30):
    
    conf = readconfig(config_filename)
    predict = Predictions(conf)
    
    # time1 = 235959
    # time2 = 2    
    # print(time1, time2, predict.addTimes(time1, time2))

    # time1 = 235939
    # time2 = 30    
    # print(time1, time2, predict.addTimes(time1, time2))

    # time1 = 21023
    # time2 = 5030    
    # print(time1, time2, predict.addTimes(time1, time2))
    
    # time1 = 21053
    # time2 = 30    
    # print(time1, time2, predict.addTimes(time1, time2))
    
    predictionsTrapPath = csvPath + trap + '/'
    threshold=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    allPredictions = []
    for csvFile in sorted(os.listdir(predictionsTrapPath)):
        if csvFile.endswith('.csv'):
            predictionFile = predictionsTrapPath + csvFile
            print(predictionFile)
            predicted = predict.load_predictions(predictionFile, filterTime=0, threshold=threshold)
            predictedSelect = predict.select_predictions(predicted, sampleTime)
            allPredictions += predictedSelect
            #print(predictedSelect)
    
    print(len(allPredictions))

    
    # %% Insect plots
if __name__ == '__main__':

    countsTh = 2 # 4 sec or three detections in one track
    percentageTh = 50  
    plt.rcParams.update({'font.size': 14})
    
    #testSelectPredictions("OH1")
    
    # %% Abundance plots
    plotAbundanceSelectedClasses(countsTh, percentageTh)
    
    traps = ['LV1', 'LV2', 'LV3', 'LV4', 'OH1', 'OH2', 'OH3', 'OH4', 'SS1', 'SS2', 'SS3', 'SS4']
    for trap in traps:
        plotAbundanceAllClasses(trap, countsTh, percentageTh, "./abundance/" + trap +"_Abundance.png")
    for trap in traps:
        plotAbundanceAllClasses(trap, countsTh, percentageTh, "./abundanceSnap/" + trap +"_Abundance.png", useSnapImages=True)
    
    # %% Tracs vs. snap plots
    plotTracksVsSnap(countsTh, percentageTh)
    
    # %% Time histogram plots
    traps = ['OH1', 'OH2', 'OH3', 'OH4']
    plotTimeHistogramsSelectedTrap(traps, countsTh, percentageTh, "OH")
    
    
    
    
    #df = pd.read_json(trackPath + '20220808TR.json')
    # Opening JSON file
    #f = open(trackPath + '20220808TR.json')
    #f = open(trackPath + 'statistics.json')
     
    # returns JSON object as 
    # a dictionary
    #data = json.load(f)

    #print(df.to_string()) 
    #df = pd.read_json(trackPath + 'statistics.json')
    #print(df.to_string()) 

