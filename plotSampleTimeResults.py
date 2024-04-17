# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:35:12 2024

@author: Kim Bjerge
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

labelNamesPlot = ["Araneae", "Coleoptera", "Diptera Brachycera", "Diptera Nematocera", "Diptera Tipulidae", 
                  "Diptera Trichocera", "Ephemeroptera", "Hemiptera", "Hymenoptera Other", "Hymenoptera Vespidae", 
                  "Lepidoptera Macros", "Lepidoptera Micros", "Neuroptera", "Opiliones", "Trichoptera"]

labelNamesPlot1 =  ["Araneae", "Coleoptera", "Diptera Brachycera", "Diptera Nematocera", "Diptera Tipulidae"]
labelNamesPlot2 =  ["Diptera Trichocera", "Ephemeroptera", "Hemiptera", "Hymenoptera Other", "Hymenoptera Vespidae"] 
labelNamesPlot3 = ["Lepidoptera Macros", "Lepidoptera Micros", "Neuroptera", "Opiliones", "Trichoptera"]
labelNamesLepidoptera = ["Lepidoptera Macros", "Lepidoptera Micros"]

#sampleTimesCorrelation.append([sampleTime, len(dayOfYear), sum(abundanceTrack), sum(abundanceTL), correlation, cosine])

id_sampleTime = 0
id_numDays = 1
id_abundanceTrack = 2
id_abundanceTL = 3
id_correlation = 4
id_cosine = 5

def plotSampleTimeCorrelation(trap, trapCorrelations, labelNames):
    
    colorIdx = 0
    colors = ["green", "blue", "purple", "cyan", "olive", "brown", "red", "pink"]
    figure = plt.figure(figsize=(10,10))
    figure.tight_layout(pad=1.0)
    ax = figure.add_subplot(1, 1, 1) 
    
    for labelName in labelNames:
        trapCorrelation = trapCorrelations[labelName]
                
        sampleTimes=[]
        correlations=[]
        cosines=[]
        for record in trapCorrelation:
            sampleTimes.append(record[id_sampleTime])
            correlations.append(record[id_correlation])
            cosines.append(record[id_cosine])
            avgInsects = int(round(record[id_abundanceTrack]/record[id_numDays]))
        
        labelText = labelName + " (" + str(avgInsects) + ")"
        ax.plot(sampleTimes, correlations, label=labelText, color=colors[colorIdx], marker="+")
        colorIdx += 1
    
    ax.set_title("Correlation of time-lapse intervals (" + trap + ")")
    ax.legend()  
    ax.set_xlabel('Sample time (sec)')
    ax.set_ylabel('Pearson correlation')
    
    #plt.savefig(resultFileName + "_" + labelName + ".png")
    plt.show() 
    
    
def plotSampleTimeCorrelationTraps(trapsCorr, labelNames, resultFileName):
    
    plt.rcParams.update({'font.size': 18})
    figure = plt.figure(figsize=(20,20))
    figure.tight_layout(pad=1.0)
   
    colors = ["green", "red", "purple", 
              "brown", "brown", "purple", 
              "olive",  "cyan", "orange", 
              "red",   "green", "blue", 
              "cyan", "orange", "olive"]

    idxFig = 1
    for labelName in labelNames:
   
        if "ax" in locals():
            ax = figure.add_subplot(5, 3, idxFig, sharex = ax) #, sharey = ax) 
        else:
            ax = figure.add_subplot(5, 3, idxFig) 

        colorIdx = 0
        for trap, trapCorrelations in trapsCorr:
               
            print(trap, labelName)

            trapCorrelation = trapCorrelations[labelName]
                    
            sampleTimes=[]
            correlations=[]
            cosines=[]
            for record in trapCorrelation:
                sampleTimes.append(record[id_sampleTime])
                correlations.append(record[id_correlation])
                cosines.append(record[id_cosine])
                avgInsects = int(round(record[id_abundanceTrack]/record[id_numDays]))
            
            labelText = trap + " (" + str(avgInsects) + ")"

            ax.plot(sampleTimes, correlations, label=labelText, color=colors[colorIdx], marker="+")
            colorIdx += 1
            ax.set_title(labelName)
            ax.legend()  
            ax.set_xlabel('Sample time (sec)')
            ax.set_ylabel('Pearson correlation')
        
        idxFig += 1
        
    plt.suptitle("Correlation of time-lapse intervals")
    plt.tight_layout(pad=1.0)
    plt.savefig("./results/" + resultFileName)
    plt.show() 

    plt.rcParams.update({'font.size': 12})

    # %% Insect plots
if __name__ == '__main__':
    
    resultPath = "./results/sampletimes/"
    
    trapsCorr = []
    for npySampleTimesFile in os.listdir(resultPath):
        
        if ".npy" in npySampleTimesFile and len(npySampleTimesFile) == 7:
            trap = npySampleTimesFile.split('.')[0]
            trapCorrelations = np.load(resultPath+npySampleTimesFile, allow_pickle=True).flat[0] 
            print(npySampleTimesFile)
            trapsCorr.append([trap, trapCorrelations])
            plotSampleTimeCorrelation(trap, trapCorrelations, labelNamesLepidoptera)
            #plotSampleTimeCorrelation(trap, trapCorrelations, labelNamesPlot1)
            #plotSampleTimeCorrelation(trap, trapCorrelations, labelNamesPlot2)
            #plotSampleTimeCorrelation(trap, trapCorrelations, labelNamesPlot3)
    
    plotSampleTimeCorrelationTraps(trapsCorr, labelNamesPlot, "sampleTimeCorrelation.png")
    
    