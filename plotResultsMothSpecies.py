# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:15:07 2024

@author: Kim Bjerge
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.linalg import norm
#import json
import datetime
from scipy.stats import expon
from idac.configreader.configreader import readconfig
from idac.predictions.predictions import Predictions
from scipy.stats import pearsonr

orderSuborderNames = ["Araneae", "Coleoptera", "Diptera Brachycera", "Diptera Nematocera", "Diptera Tipulidae", 
                      "Diptera Trichocera", "Ephemeroptera", "Hemiptera", "Hymenoptera Other", "Hymenoptera Vespidae", 
                      "Lepidoptera Macros", "Lepidoptera Micros", "Neuroptera", "Opiliones", "Trichoptera", "Vegetation"]

labelNamesPlot = ["Araneae", "Coleoptera", "Diptera Brachycera", "Diptera Nematocera", "Diptera Tipulidae", 
                  "Diptera Trichocera", "Ephemeroptera", "Hemiptera", "Hymenoptera Other", "Hymenoptera Vespidae", 
                  "Lepidoptera Macros", "Lepidoptera Micros", "Neuroptera", "Opiliones", "Trichoptera"]

#labelMothsPlot = ["Amphipyra pyramidea", "Autographa gamma", "Noctua fimbriata", "Xestia", "Agrotis puta", 
#                  "Sphinx ligustri", "Catocala", "Lemonia dumi", "Arctia caja", "Saturnia pavonia", 
#                  "Acherontia atropos", "Staurophora celsia", "Lomaspilis marginata", "Hyles", "Biston"]

labelMothsPlot = ["Agrotis puta", "Amphipyra pyramidea", "Arctia caja", "Autographa gamma", "Biston",
                  "Catocala", "Deltote pygarga", "Hyles", "Hypomecis", "Lomaspilis marginata", 
                  "Noctua fimbriata", "Phalera bucephala", "Spilosoma lubricipeda", "Staurophora celsia", "Xestia"]

config_filename = './ITC_config.json'

yearSelected = "2024" # Select year 2022, 2023, 2024
trackPath = "./tracks_" + yearSelected + "_moths/"
csvPath = "./CSV/M" + yearSelected + "S/"

#trackPath = "./tracks/tracks_order/"
#csvPath = "./CSV/M2022/"
#trackPath = "./tracks/tracks_060524_spieces/"

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
    
    def formatTime(self, recTime):
        minutes = self.getMinutes(recTime)
        seconds = self.getSeconds(recTime)
        text = str(minutes).zfill(2) + ":" + str(seconds).zfill(2)
        return text
    
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

def loadTrackFiles(trap, countsTh, percentageTh, trackPath=trackPath):

    trackFiles = trackPath + trap + '/'  
    dataframes = []
    for fileName in sorted(os.listdir(trackFiles)):
        if "TR.csv" in fileName:
            #print(trap, trackFiles + fileName)
            #data_df = pd.read_json(trackFiles + fileName)
            print(fileName)
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
    if "S" in csvPath:
        print("Load order and species predictions", trackSnapFile)
        predicted = predict.load_species_predictions(trackSnapFile, filterTime=0, threshold=threshold)
    else:
        print("Load order predictions", trackSnapFile)
        predicted = predict.load_predictions(trackSnapFile, filterTime=0, threshold=threshold)
    
    return predicted

def findMothSpecies(dataframe, numSpecies):
    
    mothSpecies = {}
    for index, row in dataframe.iterrows():
        className = row['class']
        if className not in orderSuborderNames:
            #print(className)
            if className in mothSpecies.keys():
                mothSpecies[className] += 1
            else:
                mothSpecies[className] = 1
    
    mothSpeciesSorted = dict(sorted(mothSpecies.items(), key=lambda item: item[1], reverse=True))
    mothSpeciesNames = []
    for i, key in enumerate(mothSpeciesSorted):
        mothSpeciesNames.append(key)
        if i >= numSpecies-1: 
            break;
   
    return mothSpeciesSorted, mothSpeciesNames

def plotMothSpecies(trap, mothSpecies, resultFileName, numSpecies, selectedYear=""):
    
    figure = plt.figure(figsize=(20,20))
    figure.tight_layout(pad=1.0)
    plt.rcParams.update({'font.size': 18})
    ax = figure.add_subplot(1,1,1)
    
    species = []
    abundance = []
    numSpeciesFive = 0
    numSpeciesAll = 0
    for i, key in enumerate(mothSpecies):
        if i < numSpecies:
            species.append(key)
            abundance.append(mothSpecies[key])
        if mothSpecies[key] > 4:
            numSpeciesFive += 1
        numSpeciesAll += 1
            
    print(trap, numSpeciesFive, numSpeciesAll)    

    #bar_labels = ['red', 'blue', '_red', 'orange']
    #bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
    
    ax.barh(species, abundance)
    
    ax.set_xlabel('Tracks')
    ax.set_title('Abundance of moth species ' + trap + ' ' + selectedYear)
    #ax.legend(title='Fruit color')
    plt.tight_layout(pad=2.0)
    plt.savefig("./results/" + resultFileName + "LT" + selectedYear + ".png")
    
    plt.show()
        
def plotAbundanceAllClasses(trap, countsTh, percentageTh, resultFileName, useSnapImages=False):

    trackFiles = trackPath + trap + '/'
    
    dateList, dayOfYear, selDataset2 = loadTrackFiles(trap, countsTh, percentageTh)
    mothSpecies, mothSpeciesNames = findMothSpecies(selDataset2, 15)
    plotMothSpecies(trap, mothSpecies, resultFileName, numSpecies=50)
  
    td = timedate()
    subtitle = trap + " (" + td.strMonthDay(dateList[0]) + '-' + td.strMonthDay((dateList[-1])) + ")"
 
    if useSnapImages:    
        predicted = loadSnapFiles(trap)

    figure = plt.figure(figsize=(20,20))
    figure.tight_layout(pad=1.0)
    plt.rcParams.update({'font.size': 20})
                          
    idxFig = 1
    labelNamesPlot = mothSpeciesNames 
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
        abundance = countAbundance(selDataset, dateList)
        print(trap, labelName, len(selDataset), sum(abundance))

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
        ax.set_xlim(dayOfYear[0], dayOfYear[-1]) # NB adjust for days operational
        if idxFig in [1, 4, 7, 10, 13]: 
            if useSnapImages:
                ax.set_ylabel('Observations')
            else:
                ax.set_ylabel('Tracks')
        
        idxFig += 1
  
    plt.suptitle(subtitle)
    plt.tight_layout(pad=2.0)
    plt.savefig("./results/" + resultFileName + ".png")
    plt.show() 

def plotAbundanceAllYears(trap, selectedYears, countsTh, percentageTh, resultFileName, useSnapImages=False):
  
    firstYear = True
    trackFiles = trackPath + trap + '/'
    
    for selectedYear in selectedYears:
        
        yearTrackPath = "./tracks_" + selectedYear + "_moths/"
        
        dateList, dayOfYear, selDataset2 = loadTrackFiles(trap, countsTh, percentageTh, trackPath=yearTrackPath)
        mothSpecies, mothSpeciesNames = findMothSpecies(selDataset2, 15)
        #plotMothSpecies(trap, mothSpecies, resultFileName, numSpecies=50, selectedYear=selectedYear)
      
        td = timedate()
        subtitle = trap + " " + selectedYear + " (" + td.strMonthDay(dateList[0]) + '-' + td.strMonthDay((dateList[-1])) + ")"
        
        print(subtitle)
     
        if useSnapImages:    
            predicted = loadSnapFiles(trap)
        

        figure = plt.figure(figsize=(20,20))
        figure.tight_layout(pad=1.0)
        plt.rcParams.update({'font.size': 20})
                              
        idxFig = 1
        if firstYear == True:
            firstYear = False
            #labelNamesPlot = mothSpeciesNames 
            labelNamesPlot = labelMothsPlot
            
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
            selDataset = selDataset2.loc[selDataset2['class'].str.contains(labelName)]
            abundance = countAbundance(selDataset, dateList)
            print(trap, labelName, len(selDataset), sum(abundance))
    
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
            ax.set_xlim(dayOfYear[0], dayOfYear[-1]) # NB adjust for days operational
            if idxFig in [1, 4, 7, 10, 13]: 
                if useSnapImages:
                    ax.set_ylabel('Observations')
                else:
                    ax.set_ylabel('Tracks')
            ax.set_xlim(120, 280)
            
            idxFig += 1
      
        plt.suptitle(subtitle)
        plt.tight_layout(pad=2.0)
        plt.savefig("./results/" + resultFileName + selectedYear + ".png")
        plt.show() 


if __name__ == '__main__':

    countsTh = 2 # 4 sec or three detections in one track
    percentageTh = 50  
    
    # %% tíme-lapse sample times vs. motion tracks
    
    #analyseSampleTime(countsTh, percentageTh)
      
    plt.rcParams.update({'font.size': 14})
    
    # %% Abundance plots
    #plotAbundanceSelectedClasses(countsTh, percentageTh)
    
    traps = ['LV1', 'LV2', 'LV3', 'LV4', 'OH1', 'OH2', 'OH3', 'OH4', 'SS1', 'SS2', 'SS3', 'SS4']
    #traps = ['SS3', 'SS4']
    #analyseSnapFiles(traps)
    
    #traps = ['LV1']
    for trap in traps:
        plotAbundanceAllClasses(trap, countsTh, percentageTh, "./abundance_moths_" + yearSelected + "/" + trap +"_Abundance") # Change year here!!!

    #plotYears = ["2022", "2023", "2024"]
    #for trap in traps:
    #    plotAbundanceAllYears(trap, plotYears, countsTh, percentageTh, "./abundance_moths_all_years/" + trap + "_")
    