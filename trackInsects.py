# -*- coding: utf-8 -*-
"""
Created on Mon April  1 08:10:16 2024

@author: Kim Bjerge
         Aarhus University
"""
import time
import os
import json
import argparse
from skimage import io
from idac.configreader.configreader import readconfig
from idac.datareader.data_reader import DataReader
from idac.tracker.tracker import Tracker
from idac.tracker.tracksSave import TracksSave
from idac.imagemod.image_mod import Imagemod
from idac.moviemaker.movie_maker import MovieMaker
from idac.predictions.predictions import Predictions

# Label names for plot
#labelNames = ["Araneae", "Coleoptera", "Brachycera", "Nematocera", "Tipulidae", "Trichocera", "Ephemeroptera", "Hemiptera",
#              "Hymenoptera", "Vespidae", "Macros", "Micros", "Neuroptera", "Opiliones", "Trichoptera", "Vegetation"]

config_filename = './ITC_config.json'

useNewPredictionsFormat=True # New format with species predictions
useSpeciesPredictions=True
if useSpeciesPredictions:
    from idac.stats.statsSpecies import Stats
else:    
    from idac.stats.stats import Stats
       
def run(imgPath, dirName, csvPath, trapName='', useFeatureVector=True):
    
    conf = readconfig(config_filename)
    conf['datareader']['datapath'] += '/' + dirName
    print(conf['datareader']['datapath'])

    if trapName != '':
        conf['moviemaker']['resultdir'] += '/' + trapName
        if os.path.exists(conf['moviemaker']['resultdir']) == False:
            os.mkdir(conf['moviemaker']['resultdir'])
            print("Result trap dir created", conf['moviemaker']['resultdir'])

    print(conf['moviemaker']['resultdir'])
    writemovie = conf['moviemaker']['writemovie']
    reader = DataReader(conf)
    gen = reader.getimage()
    print(type(gen))
    tr = Tracker(conf)
    imod = Imagemod()
    if dirName == '':
        dirName = 'tracks'
    mm = MovieMaker(conf, name=dirName + '.avi')
    stat = Stats(conf)
    predict = Predictions(conf)
    tracksFilename = conf['moviemaker']['resultdir']+'/'+dirName+'TRS.csv'
    print(tracksFilename)
    tracks = TracksSave(tracksFilename)

    csvFilename = csvPath+dirName+'.csv'
    if useFeatureVector:
        npyFilename = csvPath+dirName+'.npy'
    else:
        npyFilename = None
    threshold=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    if useNewPredictionsFormat: # New format with species predictions
        predicted = predict.load_species_predictions(csvFilename, filterTime=0, threshold=threshold, scoresFilename=npyFilename) #, endTime=1000) # Skip if not moved within 5 minutes
    else: # Old format without species predictions
        predicted = predict.load_predictions(csvFilename, filterTime=0, threshold=threshold, scoresFilename=npyFilename) #, endTime=1000) # Skip if not moved within 5 minutes
    totPredictions, totFilteredPredictions = predict.getPredictions()
    total = len(predicted)
    startid = 0

    iterCount = 0
    firstTime = 1
    oldFile = ""
    for insect in predicted:
        file = insect['image']
        if oldFile == file:
            oldFile = file
            continue
        else:
            oldFile = file
            
        iterCount += 1
        print('Image nr. ' + str(iterCount) + '/' + str(total), file)
        time1 = time.time()
        
        if firstTime == 1:
            firstTime = 0
            count, ooi1 = predict.findboxes(file, predicted, useSpeciesPredictions)
            for oi in ooi1:
                oi.id = startid
                startid = startid + 1            
        
        count2, ooi2 = predict.findboxes(file, predicted, useSpeciesPredictions)
            
        if count2 > 0:
            goods, startid = tr.track_boxes(ooi1, ooi2, count2, startid)
            ooi1 = goods
            tracks.save(insect, goods)
            stat.update_stats(goods, file)
            #print(stat.count)

            if writemovie:
                #file_name = conf['datareader']['datapath'] + '/' + file
                file_name = imgPath + dirName + '/' + file
                im = io.imread(file_name)
                image = imod.drawoois(im, goods)
                height, width, channel = image.shape
        
                # Write frame
                mm.writeframe(image, file)

            time2 = time.time()
            #print('Processing image took {:.3f} ms'.format((time2 - time1) * 1000.0))

    if writemovie:
        mm.releasemovie()
        
    tracks.close()
    resultdir = conf['moviemaker']['resultdir'] + '/'
    stat.writedetails(resultdir + dirName)

    return stat, resultdir, iterCount, totPredictions, totFilteredPredictions

def print_totals(date, stat, resultdir):
    record = str(date) + ','
    for spec in stat.species:
        if stat.count[spec] > 0:
            print(spec, stat.count[spec])
        record += str(stat.count[spec]) + ','
    print('Total', stat.count['total'])
    record += str(stat.count['total']) + '\n'

    resultStatFile = resultdir + 'statistics.csv'
    if os.path.exists(resultStatFile) == False:
        conf = readconfig(config_filename)
        labelNames = conf['classifier']['species']
        file = open(resultStatFile, 'w') 
        file.write('date')
        for name in labelNames:
            file.write(',' + name)
        file.write(',unknown,total\n')
    else:
        file = open(resultStatFile, 'a')
    file.write(record)
    file.close()

    stat.count['date'] = date
    file = open(resultdir + 'statistics.json', 'a')
    json_object = json.dumps(stat.count, indent=4)
    file.write(json_object + ',\n')
    file.close()

def trackInsectsOH3():

    dirNames = [ 
                #'20220808', 
                #'20220721', 
                #'20220722', 
                '20220723'
                ]
    
    imageCounts = 0
    totalPredictions = 0
    for dirName in dirNames:
        print(dirName)
        stat, resultdir, counts, totPred, totFiltered = run('O:/Tech_TTH-KBE/MAVJF/data/2022/OH3/', dirName, './CSV/M2022/')
        totalPredictions += totPred
        imageCounts += counts
        date = int(dirName[0:8])  # format YYYYMMDD
        print_totals(date, stat, resultdir)
        
    print("Total images", imageCounts, "Total detections", totalPredictions)
 
def trackInsects(imgPath, csvPath, trapName):
    
    imageCounts = 0
    totalPredictions = 0
    for dirNameCSV in os.listdir(csvPath):
        if dirNameCSV.endswith('.csv'):
            dirName = dirNameCSV.split('.')[0]
            print(trapName, dirName)
            stat, resultdir, counts, totPred, totFiltered = run(imgPath, dirName, csvPath, trapName=trapName)
            totalPredictions += totPred
            imageCounts += counts
            date = int(dirName[0:8])  # format YYYYMMDD
            print_totals(date, stat, resultdir)    

    print("Total images", imageCounts, "Total detections", totalPredictions)

def trackAllTraps(trapNames):
    
    for trapName in trapNames:
        #csvPath = './CSV/M2022S/' + trapName + '/'
        #imgPath = 'O:/Tech_TTH-KBE/MAVJF/data/2022/' + trapName + '/'
        csvPath = './CSV/M2024S/' + trapName + '/'
        imgPath = 'O:/Tech_TTH-KBE/MAVJF/data/2024/' + trapName + '/'
        #csvPath = './CSV/M2023S/' + trapName + '/'
        #imgPath = 'O:/Tech_TTH-KBE/MAVJF/data/2023/' + trapName + '/'
        #csvPath = './M2022S/' + trapName + '/'
        #imgPath = '/mnt/Dfs/Tech_TTH-KBE/MAVJF/data/2022/' + trapName + '/'
        trackInsects(imgPath, csvPath, trapName)

def trackInsectsSingleDate(imgPath, csvPath, dateStr):
    
    imageCounts = 0
    totalPredictions = 0
    for dirNameCSV in os.listdir(csvPath):
        if dateStr in dirNameCSV and dirNameCSV.endswith('.csv'):
            dirName = dirNameCSV.split('.')[0]
            print(csvPath, imgPath, dirName)
            stat, resultdir, counts, totPred, totFiltered = run(imgPath, dirName, csvPath)
            totalPredictions += totPred
            imageCounts += counts
            date = int(dirName[0:8])  # format YYYYMMDD
            print_totals(date, stat, resultdir)    

    print("Total images", imageCounts, "Total detections", totalPredictions)
    
if __name__ == '__main__':

    print('Tracking insects - see additional parameters in ITC_config.json')
    print('  Videos and results (*TR.CSV) are saved in directory specified by moviemaker -> resultdir')
    print('  To enable creating video with bounding boxes and track lines set movimaker -> writemovie:true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--traps', default='multiple') # Process data from a 'single' trap or a list of 'multiple' traps
    parser.add_argument('--species', default=True) # Use order/suborder classifer and moth species classifier if True (False only order/suborder)
    parser.add_argument('--date', default='20220618') # Set date for tracking - looking for <date>.csv file
    args = parser.parse_args()
    print(args)    
    
    # Tracking detections with order/suborder and moth species classification
    useSpeciesPredictions = args.species # If False then only order/suborder classifer is used
    
    # Tracking of multiple traps with detections in CSV files for each day
    if args.traps == 'multiple':
        #trapNames = ['LV1', 'LV2', 'LV3', 'LV4', 'OH1', 'OH2', 'OH3', 'OH4', 'SS1', 'SS2', 'SS3', 'SS4']
        #trapNames = ['LV1', 'LV2', 'LV3', 'LV4']
        #trapNames = ['OH1', 'OH2', 'OH3', 'OH4']
        #trapNames = ['OH1', 'OH2', 'OH4']
        #trapNames = ['LV3', 'LV4']
        #trapNames = ['SS1', 'SS2', 'SS4']
        trapNames = ['SS4', 'SS1']
        trackAllTraps(trapNames)

    # Tracking from a single trap with detections in CSV files for a specified day (args.date)
    if args.traps == 'single':        
        csvPath = './CSV/M2022S/LV2/' # Set path for where to find detections (CSV file)
        #imgPath = './images/' # Set path for where to find images - only used if "writemovie": true in ITC_config.json
        imgPath = 'O:/Tech_TTH-KBE/MAVJF/data/2022/LV2/'
        time1 = time.time()
        trackInsectsSingleDate(imgPath, csvPath, args.date)
        print("Tracking time", time.time()-time1)

    # Old testing trackInsectsOH3()
