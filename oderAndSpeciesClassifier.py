# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 11:47:05 2024

@author: Kim Bjerge
"""
import cv2
import numpy as np
from PIL import Image
import torch
import pandas as pd
from torchvision import transforms
from scipy.stats import norm
from resnet50 import ResNet50
import ml.models.classification as speciesClassifier


class orderSpeciesClassifier:
    '''ResNet-50 Architecture with pretrained weights
    '''
    def __init__(self, savedWeights, thresholdFile, device, img_size):
        
        print("Order classifier - threshold file", thresholdFile, "and weights", savedWeights, "of image size", img_size)

        data_thresholds = pd.read_csv(thresholdFile)
        self.labels = data_thresholds["ClassName"].to_list()
        self.thresholds = data_thresholds["Threshold"].to_list()
        self.means = data_thresholds["Mean"].to_list()
        self.stds = data_thresholds["Std"].to_list()
        print(self.labels)

        self.img_depth = 3
        self.img_size = img_size
        self.device = device

        # Use the AMI classifier model
        self.speciesModel = speciesClassifier.UKDenmarkMothSpeciesClassifierMixedResolution("")
    
        num_classes=len(self.labels)
        print("Use ResNet50 and load weights with num. classes", num_classes)
        
        self.orderModel = ResNet50(num_classes=num_classes) 
        self.orderModel.load_state_dict(torch.load(savedWeights, map_location=device))
        self.orderModel = self.orderModel.to(device)
        self.batch_size = 1
        self.batch_idx = 0
        self.imagesInBatch = torch.FloatTensor(self.batch_size , self.img_depth, self.img_size, self.img_size) 
           
    def createBatch(self, batch_size):
        self.batch_idx = 0
        self.batch_size = batch_size
        self.imagesInBatch = torch.FloatTensor(batch_size, self.img_depth, self.img_size, self.img_size) 
        #print("Batch created of size", self.batch_size)
        
    def appendToBatch(self, imageCrop):
        
        if self.batch_idx < self.batch_size:
            image = cv2.resize(imageCrop, (self.img_size, self.img_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = transforms.ToTensor()(image) #.unsqueeze_(0)
            self.imagesInBatch[self.batch_idx] = image
            self.batch_idx += 1
            return True
    
        return False
    
    def classifyOrderBatch(self):
                  
        self.imagesInBatch = self.imagesInBatch.to(self.device)
        predictions = self.orderModel(self.imagesInBatch)
        predictions = predictions.cpu().detach().numpy()
        predicted_labels = np.argmax(predictions, axis=1)
        
        lines = []
        for idx in range(len(predictions)):
            predicted_label = predicted_labels[idx]
            confidence_value = norm.cdf(predictions[idx][predicted_label], self.means[predicted_label], self.stds[predicted_label])
            confidence_value = round(confidence_value*10000)/100
            if predictions[idx][predicted_label] >= self.thresholds[predicted_label]:
                sure_label = True
            else:
                sure_label = False
            line = f"{self.labels[predicted_label]},{predicted_label},{confidence_value},{sure_label}"
            #print(line)
            lines.append(line)
            
        return lines, predictions

    def classifySpeciesBatch(self):
                  
        self.imagesInBatch = self.imagesInBatch.to(self.device)
        predictions = self.speciesModel.predict_batch(self.imagesInBatch)
        predictions = predictions.detach()
        predLabelsScores = self.speciesModel.post_process_batch(predictions)
        #print(predLabelsScores)

        lines = []
        for pred in predLabelsScores:
            predicted_label_text = pred[0]
            confidence_value = round(pred[1]*10000)/100
            #line = f"{predicted_label_text},{predicted_label},{confidence_value}"
            line = f"{predicted_label_text},{confidence_value}"
            #print(line)
            lines.append(line)
        
        return lines, predictions

        