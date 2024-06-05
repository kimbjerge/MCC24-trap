# MCC24-trap #
This project contains Python code for processing time-lapse and motion images from the AMI traps (detection, classification and tracking)

## Python environment files ##
envreq.txt - environment requierments

condaInstall.sh - edit file to install conda environment on Linux

raspberryInstall.sh - bash script to install Python environment on RaspberryPi 4 and 5

## Python source code files, configuration, models and scripts ##

### Images used for testing - one night of four hours recording with high activity of insects (>6,000 image)

https://drive.google.com/file/d/17ABGAg3b7hmxW4DbfI7wwZp4iU_PpbAB/view?usp=drive_link


### Insect order and species classifier files ###
ami - species classifier models

common - species classifier code

ml - species classifier code

orderClassifier.py - order classifier code 

orderAndSpeciesClassifier.py - combined order and species classifier

resnet50.py - model used for order classifier 

model_order_100524 - contains the newest ResNet50 model weights for oder classifier and thresholds with order labels


## Training and testing insect detector model ##

### YOLOv5 object detector files ###
data - YOLO configuration yaml files

models - YOLO yaml models and code

utils - YOLO source code

### Training YOLOv5 insect detector ###
trainF1.py

trainInsectsMoths.sh

### Validate YOLOv5 insect detector ###
val.py

testInsectsMoths.sh

## Detecting, classifying and tracing insects ##

### Combined YOLOv5 detection, ResNet50 order and species classifier ###
detectClassifyInsects.py - Detector and order classifier

detectClassifySpecies.py - Detector, order and species classifier

insectMoths-bestF1-1280m6.pt - YOLOv5m6 model trained to detect insects

CSV - contains CSV files with detections and npy files with features

Content of *.csv files which contain lines for each detection (YYYYMMDD.csv):

	year,trap,date,time,detectConf,detectId,x1,y1,x2,y2,fileName,orderLabel,orderId,orderConf,aboveTH,key,speciesLabel,speciesId,speciesConf

### Insect tracing ###
trackInsects.py - performs tracing of insects based on CSV files generated from combined YOLOv5 detector and ResNet50 classifier

ITC_config.json - configuration file for insect tracking

idac - source files used for insect tracking

tracks - contatins CSV and JSON files with tracks for every date (YYYMMDD*)

Content of *TR.csv files which contain lines for each track: 

	id,startdate,starttime,endtime,duration,class,counts,percentage,size,distance
 
Content of *TRS.csv files which contain lines for each detection related to track id: 

	id,key,date,time,confidence,valid,order,species,xc,yc,x1,y1,width,height,image
 
## Plotting, making movies and printing results ##
createMoveiCSV.py - Create movies based on the detection and classification CSV files without tracking

plotResults.py - Plotting results for tracking and order classifications

plotSampleTimeResults.py - Plotting results for comparing tracking and different time-lapse sampling intervals

plotStatistics.py - Calculating and printing statistics for tracking and order classifications






