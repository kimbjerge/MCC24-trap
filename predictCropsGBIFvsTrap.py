# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:11:04 2024

@author: Kim Bjerge
"""

import os
import cv2
import pandas as pd
import numpy as np
from orderAndSpeciesClassifierTrap import orderSpeciesClassifier

def cleanText(text_with_special_chars):
     
    # Initialize an empty string to store the cleaned text
    cleaned_text = ""
    # Iterate through each character in the input string
    for char in text_with_special_chars:
        # Check if the character is alphanumeric (letters or digits)
        if char.isalnum() or char == ' ' or char == '.' or char == '&' or char == ',':
            # If alphanumeric or a space, add it to the cleaned text
            cleaned_text += char
        else:
            #print("-------------------------------->Error char TracksSave", char)
            cleaned_text += ' ' # Convert special chars to space
            
    return cleaned_text
    

def processDatasetWithSpeciesClassifiers(dstCSVfile):
    
    image_path = "O:/Tech_TTH-KBE/MAVJF/Expert review of crops/Sorted_crops_species/"
    order_model = "./model_order_100524/dhc_best_128.pth"
    order_labels = "./model_order_100524/thresholdsTestTrain.csv"
    species_gbif_model = "./ami"
    species_model = "./model_species_251224/dhc_best_128.pth"
    species_labels = "./model_species_251224/thresholdsTestTrain.csv"
    device = "cpu"
    orderSpeciesClass = orderSpeciesClassifier(species_gbif_model, order_model, order_labels, species_model, species_labels, device, 128)  
    
    with open(dstCSVfile, 'w') as f:
        header = "Species,File,GBIFLabel,GBIFId,GBIFConf,TrapLabel,TrapId,TrapConf\n"
        f.write(header)
        f.close()

    for mothSpecies in sorted(os.listdir(image_path)):
        
        print(mothSpecies)
        dataDir = image_path+mothSpecies+'/'
        mothImages = sorted(os.listdir(dataDir))
        mothSpeciesImages = []
        for filename in mothImages:
            if filename.endswith('.jpg'):
                mothSpeciesImages.append(filename)
        filesTotal = len(mothSpeciesImages)
        batch_size = 32
        fileBlockIdx = 0
        
        while filesTotal > 0:
        
            if filesTotal < batch_size:
                batch_size = filesTotal
                
            print("Loading and predicting batch of files", batch_size)
            
            orderSpeciesClass.createBatch(batch_size)
            
            batchFileNames = []
            for idx in range(batch_size):
                imageCrop = cv2.imread(dataDir+mothSpeciesImages[fileBlockIdx + idx])
                orderSpeciesClass.appendToBatch(imageCrop)
                
            lineSpeciesPredictionsGBIF, _ = orderSpeciesClass.classifySpeciesBatch()
            lineSpeciesPredictionsTrap, _ = orderSpeciesClass.classifySpeciesTrapBatch()
                        
            for idx in range(batch_size):
                line = mothSpecies + ',' + mothSpeciesImages[idx]  + ',' + cleanText(lineSpeciesPredictionsGBIF[idx]) + ',' + lineSpeciesPredictionsTrap[idx] + '\n'
                with open(dstCSVfile, 'a') as f:
                    f.write(line)
                    f.close()
            
            fileBlockIdx += batch_size
            filesTotal -= batch_size    

#%% MAIN
if __name__=='__main__': 
    
    dstCSVfile = "resultGBIFvsTrap.csv"
    dstCSVacc = "resultGBIFvsTrapAcc.csv"

    #processDatasetWithSpeciesClassifiers(dstCSVfile)
    
    
    df = pd.read_csv(dstCSVfile)
    
    MothSpecies = ''
    GBIFStat = {}
    for index, row in df.iterrows():
        #print(row['Species'], row['GBIFLabel'])
        
        if row['Species'] != MothSpecies:
            MothSpecies = row['Species']
            GBIFStat[MothSpecies] = [0, 0, 0]            
    
        GBIFStat[row['Species']][0] += 1
        if row['Species'] in row['GBIFLabel']:
            GBIFStat[row['Species']][1] += 1
        if row['Species'] in row['TrapLabel']:
            GBIFStat[row['Species']][2] += 1
    
    with open(dstCSVacc, 'w') as f:
        header = "Species,GBIFAcc,TrapAcc,Support\n"
        f.write(header)
                
        for i, key in enumerate(GBIFStat):
            GBIFAcc = np.round((GBIFStat[key][1]/GBIFStat[key][0])*10000)/100
            TrapAcc = np.round((GBIFStat[key][2]/GBIFStat[key][0])*10000)/100
            Support = GBIFStat[key][0]
            print(key, GBIFAcc, TrapAcc, Support)
            line = key + ',' + str(GBIFAcc) + ',' + str(TrapAcc) + ',' + str(Support) + '\n'
            f.write(line)
    
        f.close()
    
    
    
    
    