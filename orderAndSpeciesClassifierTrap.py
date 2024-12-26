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
    def __init__(self, speciesGBIFModelPath, savedOrderWeights, thresholdOrderFile, savedSpeciesWeights, thresholdSpeciesFile, device, img_size):
        
        print("GBIF moth species model path", speciesGBIFModelPath)
        print("Trap moth species classifier - threshold file", thresholdSpeciesFile, "and weights", savedSpeciesWeights, "of image size", img_size)
        print("Order classifier - threshold file", thresholdOrderFile, "and weights", savedOrderWeights, "of image size", img_size)

        data_thresholds = pd.read_csv(thresholdOrderFile)
        self.labels = data_thresholds["ClassName"].to_list()
        self.thresholds = data_thresholds["Threshold"].to_list()
        self.means = data_thresholds["Mean"].to_list()
        self.stds = data_thresholds["Std"].to_list()
        print(self.labels)
        
        self.img_depth = 3
        self.img_size = img_size
        self.device = device

        # Load the AMI moth species classifier model, create model with image_size = 128 (Fixed in code)
        # Uncomment the model you like to use for your region of interrest
        self.speciesGBIFModel = speciesClassifier.UKDenmarkMothSpeciesClassifierMixedResolution(speciesGBIFModelPath, "")
        #self.speciesGBIFModel = speciesClassifier.QuebecVermontMothSpeciesClassifierMixedResolution(speciesGBIFModelPath, "")
        #self.speciesGBIFModel = speciesClassifier.PanamaMothSpeciesClassifierMixedResolution2023(speciesGBIFModelPath, "") # 1036 classes of species
    
        num_classes=len(self.labels)
        print("Order classifier uses ResNet50 and load weights with num. classes", num_classes)
        
        self.orderModel = ResNet50(num_classes=num_classes) 
        self.orderModel.load_state_dict(torch.load(savedOrderWeights, map_location=device))
        self.orderModel = self.orderModel.to(device)
        self.orderModel.eval()
        self.batch_size = 1
        self.batch_idx = 0
        self.imagesInBatch = torch.FloatTensor(self.batch_size , self.img_depth, self.img_size, self.img_size) 
        
        data_species_thresholds = pd.read_csv(thresholdSpeciesFile)
        self.speciesLabels = data_species_thresholds["ClassName"].to_list()
        self.speciesThresholds = data_species_thresholds["Threshold"].to_list()
        self.speciesMeans = data_species_thresholds["Mean"].to_list()
        self.speciesStds = data_species_thresholds["Std"].to_list()
        self.speciesSamples = data_species_thresholds["Samples"].to_list()
        
        num_species_classes=len(self.speciesLabels)
        print("Species classifier uses ResNet50 and load weights with num. classes", num_classes)
        
        self.speciesModel = ResNet50(num_classes=num_species_classes) 
        self.speciesModel.load_state_dict(torch.load(savedSpeciesWeights, map_location=device))
        self.speciesModel = self.speciesModel.to(device)
        self.speciesModel.eval()
        
           
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
    
    def classifySpeciesTrapBatch(self):            
        self.imagesInBatch = self.imagesInBatch.to(self.device)
        output = self.speciesModel(self.imagesInBatch)
        predictions = torch.nn.functional.softmax(output, dim=1)
        
        predictions = predictions.cpu().detach().numpy()
        predicted_labels = np.argmax(predictions, axis=1)
        
        lines = []
        for idx in range(len(predictions)):
            predicted_label = predicted_labels[idx]
            confidence_value = np.round(predictions[idx][predicted_label]*10000)/100
            
            # Below code not very accurated since insufficient samples in dataset
            #if self.speciesSamples[predicted_label] > 1: # More than one sample to estimate confidence value in percentage
            #    confidence_value = norm.cdf(predictions[idx][predicted_label], self.speciesMeans[predicted_label], self.speciesStds[predicted_label])
            #    confidence_value = round(confidence_value*10000)/100
            #else:
            #    confidence_value = 100.0
            #if predictions[idx][predicted_label] >= self.speciesThresholds[predicted_label]:
            #    sure_label = True
            #else:
            #    sure_label = False
            #line = f"{self.labels[predicted_label]},{predicted_label},{confidence_value},{sure_label}"
            
            line = f"{self.speciesLabels[predicted_label]},{predicted_label},{confidence_value}"
            #print(line)
            lines.append(line)
            
        return lines, predictions

    def classifySpeciesBatch(self):
        predictions = self.speciesGBIFModel.predict_batch(self.imagesInBatch)
        predictions = predictions.detach()
        predLabelsScores = self.speciesGBIFModel.post_process_batch(predictions)
        #print(predLabelsScores)

        lines = []
        for pred in predLabelsScores:
            predicted_label_text = pred[0]
            confidence_value = round(pred[1]*10000)/100
            predicted_label = pred[2]
            line = f"{predicted_label_text},{predicted_label},{confidence_value}"
            #print(line)
            lines.append(line)
        
        return lines, predictions

        
