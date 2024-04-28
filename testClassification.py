# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 18:53:34 2024

@author: Kim Bjerge
"""

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import ml.models.classification as classifier



def plotConfusionMatrixLevel(levelName, level_predict, level_label, labels, normalize=False, font_size=12):

    matrix = np.zeros((len(labels), len(labels))).astype('int')
    
    # Change label names
    labels = [l.replace("_ok", "") for l in labels]
    labels = [l.replace("_Identifiable to species", "") for l in labels]
    labels = [l.replace("_", " ") for l in labels]

    for i in range(len(level_predict)):
        matrix[level_label[i], level_predict[i]] += 1
              
    matrixSum = matrix.sum(axis=1)[:, np.newaxis]
    #matrixAvg = matrix.astype('float') / (matrixSum+0.001)
    matrixAvg = (100*matrix.astype('float') / (matrixSum+0.001)) + 0.5
    matrixAvg = matrixAvg.astype('int')
    
    plt.scatter(matrixSum, matrixAvg.diagonal(), marker='.', color='green')
    print(sum(matrixSum))
    plt.xlim(0,200)
    plt.ylim(10,105)
    plt.xlabel("Sample size")
    plt.ylabel("Accuracy (%)")
    plt.title("Class accuracy vs. sample size")
    
    if normalize:
        matrix = matrixAvg
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(matrix, matrixSum)        
        
    plt.rcParams.update({'font.size': font_size})
    #fig, ax = plt.subplots(figsize=(9, 7))
    fig, ax = plt.subplots(figsize=(60, 60))
    ax.imshow(matrixAvg, cmap='Greens')
    
    yLabels = []
    for idx in range(len(labels)):
        yLabels.append(labels[idx] + ' (' + str(int(matrixSum[idx,0])) + ')')
    
    ax.set(xticks=np.arange(len(labels)),
           yticks=np.arange(len(labels)),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=yLabels,
           title=levelName,
           ylabel='True label',
           xlabel='Predicted label')    
   
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    #fmt = '.2f' if normalize else 'd'
    fmt = 'd'
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = 'k'
            if i == j:
                color = 'w'
            if matrix[i, j] > 0.005:
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color=color)
    
    #ax.set_title(levelName)
    fig.tight_layout()
    plt.savefig('ConfMatrix.png')
    plt.show()  
    
    
#%% MAIN
if __name__=='__main__':
    
    img_size = 128
    img_depth = 3
    device = "cpu"
    #device = "cuda"
    #device = "cpu"
    #image_base_path = "C:/IHAK/few-shot-novelty/data/euMoths/images/"
    image_base_path = "O:/Tech_TTH-KBE/MAVJF/Training data moths classifier/MothSpecies1/"
    #image_base_path = "/mnt/Dfs/Tech_TTH-KBE/MAVJF/Training data moths classifier/MothSpecies1/"
    classModel = classifier.UKDenmarkMothSpeciesClassifierMixedResolution("")
    #labels = classifier.get_category_map()
    totalSpecies = 0
    trueSpecies = 0
    for classDir in os.listdir(image_base_path):
        if os.path.isdir(image_base_path+classDir):
 #       if os.path.isDir(classDir != '.gitignore':
            fullClassPath = image_base_path + classDir
            batch_size = 64
            
            print("============================================================")
            print("Predicting class for moth species", classDir)
            print("============================================================")
            
            #for fileName in  os.listdir(fullClassPath):
            #    print(fileName)
                
            filenames = sorted(os.listdir(fullClassPath))
            #filenames = os.scandir(fullClassPath)
            #filenames = [f for f in pathlib.Path().glob(fullClassPath+'/*')]
            filenames = [filename for filename in filenames if filename.endswith(".jpg") or filename.endswith(".JPG")]
            fileIdx = 0
            filesTotal = len(filenames)
            if filesTotal < batch_size:
                batch_size = filesTotal
            while filesTotal > 0:
                print("Loading and predicting batch of files", batch_size)
                imagesInBatch = torch.FloatTensor(batch_size, img_depth, img_size, img_size) 
                batchFileNames = []
                for idx in range(batch_size):
                    filenam = fullClassPath+ '/' +filenames[fileIdx]
                    #filenam = filenam.replace('Â', '?')
                    #print(filenam)
                    image = Image.open(filenam)
                    image = image.resize((img_size, img_size))
                    image = transforms.ToTensor()(image) #.unsqueeze_(0)
                    imagesInBatch[idx] = image
                    batchFileNames.append(filenames[fileIdx])
                    fileIdx += 1
                    
                imagesInBatch = imagesInBatch.to(device)
                predictions = classModel.predict_batch(imagesInBatch)
                predictions = predictions.detach()
                #print(predictions)
                predLabelsScores = classModel.post_process_batch(predictions)
                #print(predLabelsScores)
                for pred in predLabelsScores:
                    correct = False
                    totalSpecies += 1
                    #classDir = classDir.replace('_', ' ')
                    #classDir = classDir[0].upper() + classDir[1:]
                    if classDir in pred[0]:
                        correct = True
                        trueSpecies += 1
                    print(pred[0], pred[1], correct)
                #predictions = predictions.cpu().detach().numpy()
                #predicted_labels = np.argmax(predictions, axis=1)    
        
                filesTotal -= batch_size
                if filesTotal < batch_size:
                    batch_size = filesTotal # Use smaller size for last batch
                    
            print("Total", totalSpecies, "true", trueSpecies, "accuracy", 100*trueSpecies/totalSpecies)
    #batch_output = classModel.predict_batch(batch_input)
    #batch_output = list(classModel.post_process_batch(batch_output))
                
    #classModel = UKDenmarkMothSpeciesClassifierMixedResolution()
    
