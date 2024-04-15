# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:11:04 2024

@author: Kim Bjerge
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
import pandas as pd
from torchvision import transforms
from scipy.stats import norm
from runtime_args_moths import args
from resnet50 import ResNet50
#from efficientnet import EfficientNetBx

# def softmax(vector):
#     e = np.exp(vector)
#     return e / e.sum()

# def gaussian(x, mean, std):
#     Ex = 0.5*(((x - mean)/std)**2) 
#     A = 1/(std*np.sqrt(2*np.pi))
#     return round(A*np.exp(-Ex)*10000)/10000
    
#%% MAIN
if __name__=='__main__':
    
    #dataDir = "O:/Tech_TTH-KBE/MAVJF/dataCrops/LV1/crops/"
    dataDir = args.detect_path + args.trap + "/crops/" # Path for M
    #dataDir = args.detect_path + args.trap + "/"
    outputFile = "./results/" + args.trap + ".csv"
    savedDir = args.model_save_path + "/"
    print("Processing data in", dataDir, "saving result in", outputFile)
    
    #thresholdFile = savedDir + 'thresholdsTrain.csv'
    thresholdFile = savedDir + 'thresholdsTestTrain.csv'
    data_thresholds = pd.read_csv(thresholdFile)
    labels = data_thresholds["ClassName"].to_list()
    thresholds = data_thresholds["Threshold"].to_list()
    means = data_thresholds["Mean"].to_list()
    stds = data_thresholds["Std"].to_list()
        
    #device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    device = torch.device("cuda:1" if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')

    print("Using threshold file", thresholdFile, "and model file", savedDir+args.weights)

    num_classes=len(labels)
   # if args.model == "EfficientNetB3":
   #     model = EfficientNetBx(num_classes=num_classes, eff_name='b3')
   #     print("Use EfficientNetB3 and load weights")
   # else:
   #     if args.model == "EfficientNetB4":
   #         model = EfficientNetBx(num_classes=num_classes, eff_name='b4')
   #         print("Use EfficientNetB4 and load weights")
   #     else:
    model = ResNet50(num_classes=num_classes) 
    print("Use ResNet50 and load weights")

    model.load_state_dict(torch.load(savedDir+args.weights, map_location=device))
    
    model = model.to(device)
    
    novelClassIdx = len(thresholds)
    labels.append("Unsure") # Add unsure label as last index
    resultFile = open(outputFile, mode='w')
    resultFile.write("Filename,ClassName,ClassIdx,Probability,AboveThreshold\n")
    print("Listing files in", dataDir)
    filenames = sorted(os.listdir(dataDir))
    #filenames = [filename for filename in filenames if filename.endswith(".jpg")]
    filenames = [filename for filename in filenames if filename.endswith(".JPG") or filename.endswith(".jpg")]
    fileIdx = 0
    filesTotal = len(filenames)
    batch_size = args.batch_size
    if filesTotal < batch_size:
        batch_size = filesTotal
    while filesTotal > 0:
        print("Loading and predicting batch of files", batch_size)
        imagesInBatch = torch.FloatTensor(batch_size, args.img_depth, args.img_size, args.img_size) 
        batchFileNames = []
        for idx in range(batch_size):
            image = cv2.imread(dataDir+filenames[fileIdx])
            image = cv2.resize(image, (args.img_size, args.img_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = transforms.ToTensor()(image) #.unsqueeze_(0)
            imagesInBatch[idx] = image
            batchFileNames.append(filenames[fileIdx])
            fileIdx += 1
            
        imagesInBatch = imagesInBatch.to(device)
        predictions = model(imagesInBatch)
        predictions = predictions.cpu().detach().numpy()
        predicted_labels = np.argmax(predictions, axis=1)

        # Outlier detector
        idx = 0
        # Check for unsure predicitons and compute probability
        for idx in range(len(predictions)):
            predicted_label = predicted_labels[idx]
            #confidence_value = softmax(predictions[idx])[predicted_label]
            #confidence_value = gaussian(predictions[idx][predicted_label], means[predicted_label], stds[predicted_label])
            confidence_value = norm.cdf(predictions[idx][predicted_label], means[predicted_label], stds[predicted_label])
            confidence_value = round(confidence_value*10000)/100
            if predictions[idx][predicted_label] >= thresholds[predicted_label]:
                sure_label = True
            else:
                sure_label = False
            line = f"{batchFileNames[idx]},{labels[predicted_label]},{predicted_label},{confidence_value},{sure_label}\n"
            print(line)
            resultFile.write(line)
                
        resultFile.flush()
        
        filesTotal -= batch_size
        if filesTotal < args.batch_size:
            batch_size = filesTotal # Use smaller size for last batch
        print("Remaining files", filesTotal)
    
    resultFile.close()
    
    
