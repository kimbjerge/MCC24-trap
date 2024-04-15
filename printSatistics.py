# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:33:49 2024

@author: Kim Bjerge
"""

import os
import pandas as pd
import numpy as np
from idac.configreader.configreader import readconfig
from idac.predictions.predictions import Predictions

thresholds=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
config_filename = './ITC_config.json'

def countPredictions(predictions):
    
    filesCnt = 0
    predictionCnt = 0
    predictionAbove = 0
    predictionBelow = 0
    currentFileName = ""
    currentDay = 0
    for insect in predictions:
        if currentFileName != insect['image']:
            currentFileName = insect['image']
            filesCnt += 1
        predictionCnt += 1
        if insect['valid']:
            predictionAbove += 1
        else:
            predictionBelow += 1
    
    return filesCnt, predictionCnt, predictionAbove, predictionBelow
            
        
def printPredictionStatistics(trapNames, predict):
    
    predictionsPath = "./CSV/M2022/"
    fileCntTotal = 0
    predictionCntTotal = 0
    predictionAboveTotal = 0
    nightsTotal = 0
    for trap in trapNames:
        predictionPath = predictionsPath + trap + '/'
        
        fileCntTrap = 0
        predictionCntTrap = 0
        predictionAboveTrap = 0
        nightsTrap = 0
        for predictionFile in sorted(os.listdir(predictionPath)):
            if predictionFile.endswith('.csv'):        
                predicted = predict.load_predictions(predictionPath+predictionFile, filterTime=0, threshold=thresholds)
                filesCnt, predictionCnt, predictionAbove, predictionBelow = countPredictions(predicted)
                nightsTrap += 1
                fileCntTrap += filesCnt
                predictionCntTrap += predictionCnt
                predictionAboveTrap += predictionAbove
                nightsTotal += 1
                fileCntTotal += filesCnt
                predictionCntTotal += predictionCnt
                predictionAboveTotal += predictionAbove
        
        print(trap, nightsTrap, fileCntTrap, predictionCntTrap, 100*predictionAboveTrap/predictionCntTrap)
                
    print("Nights", nightsTotal, "files total", fileCntTotal, "predictions", predictionCntTotal,  "percentage above", 100*predictionAboveTotal/predictionCntTotal)


def printPredictionSnapStatistics(trapNames, predict):
    
    predictionsPath = "./CSV/M2022/"
    fileCntTotal = 0
    predictionCntTotal = 0
    predictionAboveTotal = 0
    for trap in trapNames:
        predictionPath = predictionsPath + 'snap' + trap + '.csv'
        
        predicted = predict.load_predictions(predictionPath, filterTime=0, threshold=thresholds)
        fileCntTrap, predictionCntTrap, predictionAboveTrap, predictionBelowTrap = countPredictions(predicted)
        fileCntTotal += fileCntTrap
        predictionCntTotal += predictionCntTrap
        predictionAboveTotal += predictionAboveTrap
        
        print(trap, fileCntTrap, predictionCntTrap, 100*predictionAboveTrap/predictionCntTrap)
                
    print("files total", fileCntTotal, "predictions", predictionCntTotal,  "percentage above", 100*predictionAboveTotal/predictionCntTotal)


def printTrackStatistics(trapNames, countsTh, percentageTh):
    
    trackPath = "./tracks/"
    
    tracksTotal = 0
    tracksValidTotal = 0
    for trap in trapNames:
        trackFiles = trackPath + trap + '/'
        dataframes = []
        for fileName in sorted(os.listdir(trackFiles)):
            if "TR.csv" in fileName:
                data_df = pd.read_csv(trackFiles + fileName)
                dataframes.append(data_df)

        dataset = pd.concat(dataframes)
        tracksTrap = len(dataset)
        selDataset1 = dataset.loc[dataset['percentage'] > percentageTh]
        selDataset2 = selDataset1.loc[selDataset1['counts'] >= countsTh]
        tracksValidTrap = len(selDataset2)
        print(trap, tracksTrap, tracksValidTrap, 100*tracksValidTrap/tracksTrap)
        tracksTotal += tracksTrap
        tracksValidTotal += tracksValidTrap
        
    print("Tracks total", tracksTotal, "valid", tracksValidTotal, "pecentage valid", 100*tracksValidTotal/tracksTotal)
    
        
#%% MAIN
if __name__=='__main__':
    
    conf = readconfig(config_filename)
    predict = Predictions(conf)
    
    trapNames = ['LV1', 'LV2', 'LV3', 'LV4', 'OH1', 'OH2', 'OH3', 'OH4', 'SS1', 'SS2', 'SS3', 'SS4']
    
    printPredictionStatistics(trapNames, predict)
    printPredictionSnapStatistics(trapNames, predict)
    printTrackStatistics(trapNames, 2, 50)
    