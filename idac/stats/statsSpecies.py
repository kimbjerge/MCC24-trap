import numpy as np
import math
import json
import pandas as pd
from datetime import datetime
from datetime import timedelta

class Stats:
    def __init__(self, config):
        self.species = config["classifier"]["species"]
        
        if config["classifier"]["speciesClassifer"] == "JSON": # Use UK/Denmark AMI species classifier
            with open(config["classifier"]["speciesJSON"]) as f:
                speciesLabels = json.load(f)
            for label in speciesLabels.keys():
                self.species.append(label)
        
        if config["classifier"]["speciesClassifer"] == "CSV": # Use Denmark AMT species classifier
            speciesLabels = pd.read_csv(config["classifier"]["speciesCSV"])
            for i in speciesLabels.index:
                self.species.append(speciesLabels.loc[i]["ClassName"])
        #print(self.species)

        self.species.append('unknown')
        self.mincounts = config["stats"]["mincounts"]
        self.count = {}
        self.count['date'] = 0
        for spec in self.species:
            self.count[spec] = 0
        self.count['unknown'] = 0
        self.count['total'] = 0
        self.idhistory = {}
        self.details = {}

    def update_stats(self, oois, imname):
        #print(imname)
        #Find info in filename
        #ind = imname.index('-')
        startdate =  imname[0:8]
        time = imname[8:14]
        time = ':'.join(time[i:i+2] for i in range(0, len(time), 2))

        for obj in oois:
            if obj.id in self.idhistory:
                obj.endtime = time
                self.calc_details(obj)
                self.idhistory[obj.id][0] += 1
                self.label_select(obj)
                if self.idhistory[obj.id][0] % (self.mincounts+1) == 0: # KBE 5
                    #count down old
                    if self.idhistory[obj.id][2] != '': # No last label
                        self.count[self.idhistory[obj.id][2]] -= 1
                    #count up new
                    self.count[obj.label] += 1
                    if self.idhistory[obj.id][0] == (self.mincounts+1): # KBE 5
                        self.count['total'] += 1
                    self.idhistory[obj.id][2] = obj.label
            else:
                # New idhistory = [number of detections, index array of species, name of classification]
                self.idhistory[obj.id] = [1, np.zeros(len(self.species)), '']
                obj.starttime = time
                obj.endtime = time
                obj.startdate = startdate
                obj.count = 1
                self.calc_details(obj)
            arr = self.idhistory[obj.id][1]
            unknown_pos = len(arr)-1
            total_count = np.sum(arr[:unknown_pos])
            total_count = total_count + arr[unknown_pos] * 2
            obj.counts = total_count
            obj.boxsizehist.append(obj.w*obj.h)

    def label_select(self, obj):

        if obj.label not in self.species: # KBE??? Check if label in species list
            print("Unknown class ", obj.label)
            obj.label = 'unknown'

        if obj.label == 'unknown':
            weight = 0.5
        else:
            weight = 1
            
        index = self.species.index(obj.label)                      
        self.idhistory[obj.id][1][index] += weight
        index = np.argmax(self.idhistory[obj.id][1])
        obj.label = self.species[index]

    def calc_details(self, obj):
        pos_iter = iter(obj.centerhist)
        prev_position = next(pos_iter)
        distance = 0
        if len(obj.centerhist) > 1:
            for pos in pos_iter:
                distance += int(math.sqrt(((pos[0] - prev_position[0]) ** 2) + ((pos[1] - prev_position[1]) ** 2)))
                prev_position = pos
        #print("Distance is: " + str(distance))
        self.details[obj.id] = obj
        obj.distance = distance

    def cleanText(self, text_with_special_chars):
        
        # Initialize an empty string to store the cleaned text
        cleaned_text = ""
        # Iterate through each character in the input string
        for char in text_with_special_chars:
            # Check if the character is alphanumeric (letters or digits)
            if char.isalnum() or char == ' ' or char == '.' or char == '&' or char == '-':
                # If alphanumeric or a space, add it to the cleaned text
                cleaned_text += char
            else:
                #print("-------------------------------->Error char statsSpecies", char)
                cleaned_text += ' ' # Convert special chars to space
                
        return cleaned_text
    
    def writedetails(self, dirname):
        file = open(dirname + 'TR.json', 'w+')
        line = '{\n\"tracks\":[\n'
        file.write(line)
        filecsv = open(dirname + 'TR.csv', 'w+')
        line = 'id,startdate,starttime,endtime,duration,class,counts,percentage,size,distance\n'
        filecsv.write(line)
 
        firstLine=True
        for key in self.details.keys():
            obj = self.details[key]
            #Calculate confidence
            ind = self.species.index(obj.label)
            counts = obj.counts
            if counts == 0: 
                counts = 1
            if obj.label == 'unknown':
                conf = (self.idhistory[obj.id][1][ind]*2) / (counts + 1)
            #elif counts > 0:
            else:
                conf = self.idhistory[obj.id][1][ind] / (counts + 1)

            #Distance
            distance = obj.distance

            #Calculate duration
            s1 = obj.starttime
            s2 = obj.endtime  # for example
            FMT = '%H:%M:%S'
            tdelta = datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)
            if tdelta.days < 0:
                tdelta = timedelta(days=0, seconds=tdelta.seconds, microseconds=tdelta.microseconds)
            tdelta_seconds = tdelta.total_seconds()

            #calculate avg blob size
            avg_blob = np.mean(obj.boxsizehist)

            #Format string
            if obj.counts >= self.mincounts: #JBN??? 4 should be same threshold as for statistic
                if firstLine:
                    firstLine=False
                else:
                    file.write(',\n')
                    
                towrite = '{\"id\":' + str(obj.id) + ','  + '\"startdate\":' + obj.startdate + ',' + '\"starttime\":\"' + obj.starttime + '\",' + '\"endtime\":\"' + obj.endtime + '\",' \
                          + '\"duration\":' + str(int(tdelta_seconds)) + ',' + '\"class\":' + '\"' + self.cleanText(obj.label) + '\",' \
                          + '\"counts\":' + str(int(obj.counts)) + ',' + '\"percentage\":' + "%0.2f" % (conf*100) + ',' + '\"size\":' \
                          + "%0.2f" % avg_blob + ',' + '\"distance\":' + str(int(distance)) + '}'
                file.write(towrite)
                
                line = str(obj.id) + ',' + obj.startdate + ',' + obj.starttime + ',' + obj.endtime + ',' + str(int(tdelta_seconds)) + ',' \
                       + self.cleanText(obj.label) + ',' + str(int(obj.counts)) + ',' + "%0.2f" % (conf*100) + ','  + "%0.2f" % avg_blob + ',' + str(int(distance)) + '\n'
                filecsv.write(line)

        line = ']\n}\n'
        file.write(line)
        file.close()
        filecsv.close()
        

