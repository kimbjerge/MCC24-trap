# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:35:12 2024

@author: Kim Bjerge
"""
import os
import math
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
# Correction of label names in paper
def correctLabelName(name):
    
    if "Lepidoptera Macros" in name:
        name = name.replace("Macros", "Macro")
    if "Lepidoptera Micros" in name:
        name = name.replace("Micros", "Micro")
    if "Diptera Trichocera" in name:
        name = name.replace("Trichocera", "Trichoceridae")
        
    return name

id_sampleTime = 0
id_numDays = 1
id_abundanceTrack = 2
id_abundanceTL = 3
id_correlation = 4
id_cosine = 5 # Could also be avgSumAbsDiff

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
        
        labelText = correctLabelName(labelName) + " (" + str(avgInsects) + ")"
        ax.plot(sampleTimes, correlations, label=labelText, color=colors[colorIdx], marker="+")
        colorIdx += 1
    
    ax.set_title("Correlation of tracks vs. TL sampling intervals (" + trap + ")")
    ax.legend()  
    ax.set_xlabel('Sample time (sec)')
    ax.set_ylabel('Pearson correlation')
    
    #plt.savefig(resultFileName + "_" + labelName + ".png")
    plt.show() 
    
    
def plotSampleTimeCorrelationTraps(trapsCorr, labelNames, resultFileName, usePearson=True, useSampleTimes=True, simSampleTime=100):
    
    plt.rcParams.update({'font.size': 18})
    figure = plt.figure(figsize=(20,20))
    figure.tight_layout(pad=1.0)
   
    colors = ["green", "blue", "purple", 
              "olive", "cyan", "brown", 
              "pink",  "orange", "blue", 
              "green", "grey", "black", 
              "olive", "cyan", "brown"]

    idxFig = 1
    bestSampleTimes = []
    minAvgSumAbsDiffTimes = []
    testSampleTimes = [10, 30, 100, 200, 500, 1000, 1500, 2000, 3000]
    testSampleSeconds = [10, 30, 60, 120, 300, 600, 900, 1200, 1800]
    numTestSampleTimes =[0, 0,   0,   0,   0,    0,    0,    0,    0]
    numDiffSampleTimes =[0, 0,   0,   0,   0,    0,    0,    0,    0]
    for labelName in labelNames:
   
        if "ax" in locals():
            if useSampleTimes:
                ax = figure.add_subplot(5, 3, idxFig, sharex = ax) #, sharey = ax) 
            else:
                ax = figure.add_subplot(5, 3, idxFig) 
        else:
            ax = figure.add_subplot(5, 3, idxFig) 

        colorIdx = 0
        sumInsects = 0
        sumDays = 0
        correlationAllTraps = []
        abundanceAllTraps = []
        correlationAllTimes = []
        differenceAllTimes = []
        for trap, trapCorrelations in trapsCorr:
               
            print(trap, labelName)

            trapCorrelation = trapCorrelations[labelName]
                    
            sampleTimes=[]
            correlations=[]
            cosines=[]
            for record in trapCorrelation:
                if record[id_sampleTime] == simSampleTime or useSampleTimes:
                    sampleTimes.append(record[id_sampleTime])
                    correlations.append(record[id_correlation])
                    if usePearson:
                        correlationAllTraps.append(record[id_correlation])
                    else:
                        correlationAllTraps.append(record[id_cosine])
                    cosines.append(record[id_cosine])
                    sumInsects += record[id_abundanceTrack]
                    sumDays += record[id_numDays]
                    avgInsects = int(round(record[id_abundanceTrack]/record[id_numDays]))
                    avgInsectsTL = round(record[id_abundanceTL]/record[id_numDays]*100)/100
                    abundanceAllTraps.append(avgInsectsTL)
            
            if not any([math.isnan(c) for c in cosines]):
                differenceAllTimes.append(cosines)
            if not any([math.isnan(c) for c in correlations]):
                correlationAllTimes.append(correlations)
            maxCorr = max(correlations)
            idxMaxCorr = correlations.index(maxCorr)
            bestSampleTimes.append(sampleTimes[idxMaxCorr])
            idx = testSampleTimes.index(sampleTimes[idxMaxCorr])
            numTestSampleTimes[idx] += 1
            avgInsectsTraps = round((sumInsects/sumDays)*10)/10
            
            minError = min(cosines) # avgSumAbsDiff
            idxMinError = cosines.index(minError)
            minAvgSumAbsDiffTimes.append(sampleTimes[idxMinError])
            idx = testSampleTimes.index(sampleTimes[idxMinError])
            numDiffSampleTimes[idx] += 1      
            
            #sampleTimes = [time/100 for time in sampleTimes]
            plotSampleTimes = []
            for time in sampleTimes:
                if time > 60:
                    time = time/100
                else:
                    time = time/60
                plotSampleTimes.append(time)

            print(plotSampleTimes)
            print(correlations)

            labelText = trap + " (" + str(avgInsects) + ")"

            if useSampleTimes:
                
                if usePearson:
                    #ax.plot(plotSampleTimes, correlations, label=labelText, color=colors[colorIdx], marker=".")
                    ax.scatter(plotSampleTimes, correlations, label=labelText, color=colors[colorIdx], marker="o")
                    ax.set_ylim(0,1)
                else:
                    ax.scatter(plotSampleTimes, cosines, label=labelText, color=colors[colorIdx], marker="o")
                    ax.set_yscale('log')
                #ax.scatter(plotSampleTimes[idxMaxCorr], maxCorr, color="black", marker="s")
                colorIdx += 1
                titleColor = "green"
                if avgInsectsTraps < 50:
                    titleColor = "blue"
                if avgInsectsTraps < 10:
                    titleColor = "red"
                                    
                ax.set_title(correctLabelName(labelName) + " (" + str(avgInsectsTraps) + ")", color=titleColor)
                #if idxFig == 1:
                #ax.legend(loc='lower right')
                #ax.set_xscale('log')
                ax.set_xlim(0,30)
                ax.set_xscale('log')
                if idxFig in [13, 14, 15]: 
                    ax.set_xlabel('Interval (minutes)')
                if idxFig in [1, 4, 7, 10, 13]: 
                    if usePearson:
                        ax.set_ylabel('Pearson correlation')
                    else:
                        #ax.set_ylabel('Cosine similarity')
                        ax.set_ylabel('Difference')

        if useSampleTimes:
            
            if usePearson:
                ax.plot(plotSampleTimes, np.mean(correlationAllTimes, axis=0), label="", color="red", marker=".")
            else:
                ax.plot(plotSampleTimes, np.mean(differenceAllTimes, axis=0), label="", color="red", marker=".")
                
        else:
            
            ax.scatter(abundanceAllTraps, correlationAllTraps, color="black")
            titleColor = "green"
            if avgInsectsTraps < 50:
                titleColor = "blue"
            if avgInsectsTraps < 10:
                titleColor = "red"
                                
            ax.set_title(correctLabelName(labelName) + " (" + str(avgInsectsTraps) + ")", color=titleColor)
            ax.set_ylim(0,1)
            if idxFig in [13, 14, 15]: 
                ax.set_xlabel('Average detections per. night')
            if idxFig in [1, 4, 7, 10, 13]: 
                if usePearson:
                    ax.set_ylabel('Pearson correlation')
                else:
                    #ax.set_ylabel('Cosine similarity')
                    ax.set_ylabel('Difference')
            
            
        idxFig += 1
        
    if useSampleTimes:
        if usePearson:
            plt.suptitle("Correlation of tracks vs. TL sampling intervals for all traps")
        else:
            plt.suptitle("Difference of tracking vs. TL sampling for all traps")
            
    else:
        if simSampleTime < 60:
            plt.suptitle("Correlation vs. average detections for all traps (TL " + str(int(simSampleTime%100)) +" sec)")
        else:    
            plt.suptitle("Correlation vs. average detections for all traps (TL " + str(int(simSampleTime/100)) +" min)")
        
    plt.tight_layout(pad=2.0)
    plt.savefig("./results/" + resultFileName)
    plt.show() 

    plt.rcParams.update({'font.size': 12})
    return bestSampleTimes, testSampleTimes, testSampleSeconds, numTestSampleTimes, numDiffSampleTimes

    # %% Insect plots
if __name__ == '__main__':
    
    resultPath = "./results/sampletimes/"
    
    #traps = ['LV1', 'LV2', 'LV3', 'LV4', 'OH1', 'OH2', 'OH3', 'OH4', 'SS1', 'SS2', 'SS3', 'SS4']
    traps = ['LV1', 'LV2', 'LV3', 'LV4', 'OH1', 'OH2', 'OH3', 'OH4', 'SS1', 'SS2', 'SS4']
    trapsCorr = []
    for npySampleTimesFile in sorted(os.listdir(resultPath)):
        
        if ".npy" in npySampleTimesFile and len(npySampleTimesFile) == 7:
            trap = npySampleTimesFile.split('.')[0]
            if trap in traps:
                trapCorrelations = np.load(resultPath+npySampleTimesFile, allow_pickle=True).flat[0] 
                print(npySampleTimesFile)
                trapsCorr.append([trap, trapCorrelations])
                #plotSampleTimeCorrelation(trap, trapCorrelations, labelNamesLepidoptera)
                #plotSampleTimeCorrelation(trap, trapCorrelations, labelNamesPlot1)
                #plotSampleTimeCorrelation(trap, trapCorrelations, labelNamesPlot2)
                plotSampleTimeCorrelation(trap, trapCorrelations, labelNamesPlot3)
    
    
    bestSampleTimes, testSampleTimes, testSampleSeconds, numTestSampleTimes, numDiffSampleTimes = plotSampleTimeCorrelationTraps(trapsCorr, labelNamesPlot, "sampleTimeCorrelation2min.png", useSampleTimes=False, simSampleTime=200)
    bestSampleTimes, testSampleTimes, testSampleSeconds, numTestSampleTimes, numDiffSampleTimes = plotSampleTimeCorrelationTraps(trapsCorr, labelNamesPlot, "sampleTimeCorrelation10min.png", useSampleTimes=False, simSampleTime=1000)
    bestSampleTimes, testSampleTimes, testSampleSeconds, numTestSampleTimes, numDiffSampleTimes = plotSampleTimeCorrelationTraps(trapsCorr, labelNamesPlot, "sampleTimePearsonCorrelation.png", usePearson=True)
    bestSampleTimes, testSampleTimes, testSampleSeconds, numTestSampleTimes, numDiffSampleTimes = plotSampleTimeCorrelationTraps(trapsCorr, labelNamesPlot, "sampleTimeMeanAbsDifference.png", usePearson=False)
    
    print(bestSampleTimes)
    #plt.hist(bestSampleTimes, bins=250)
    print("Median of best sample times", np.median(bestSampleTimes))
    
    sampleTimes = []
    for time in testSampleTimes:
        if time > 60:
            time = time/100
        else:
            time = time/60
        sampleTimes.append(time)
    
    plt.rcParams.update({'font.size': 16})
    figure = plt.figure(figsize=(14,8))
    figure.tight_layout(pad=1.0)
    ax = figure.add_subplot(1, 2, 1) 
    ax.stem(sampleTimes, numTestSampleTimes)
    #plt.text(1, 40, "10s", fontsize = 14)
    #plt.text(3, 25, "2m", fontsize = 14)

    ax.set_title("Best correlation")
    ax.set_xlabel("TL interval (minutes)")
    ax.set_ylabel("Frequency")

    ax = figure.add_subplot(1, 2, 2) 
    ax.stem(sampleTimes, numDiffSampleTimes)
    #plt.text(1, 40, "10s", fontsize = 14)
    #plt.text(3, 25, "2m", fontsize = 14)

    ax.set_title("Minimum difference")
    ax.set_xlabel("TL interval (minutes)")
    #ax.set_ylabel("Frequency")
    plt.show()
    
    