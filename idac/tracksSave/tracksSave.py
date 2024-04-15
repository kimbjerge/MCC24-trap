#import numpy as np
#import math
#from datetime import datetime
#from datetime import timedelta

class TracksSave:
    
    def __init__(self, fileName):
        self.tracksFileName = fileName
        self.tracksFile = open(fileName, 'w')
        headerline = "id,key,date,time,confidence,valid,class,xc,yc,x1,y1,width,height,image\n"
        self.tracksFile.write(headerline) 
 
    def save(self, predict, ois):
        for oi in ois:
            line = str(oi.id) + ',' + \
                   str(oi.key) + ',' + \
                   str(predict['date']) + ',' + \
                   str(predict['time']) + ',' + \
                   str(oi.percent) + ',' + \
                   str(oi.valid) + ',' + \
                   oi.label  + ',' + \
                   str(int(oi.x + oi.w / 2)) + ',' + \
                   str(int(oi.y + oi.h / 2)) + ',' + \
                   str(oi.x) + ',' + \
                   str(oi.y) + ',' + \
                   str(oi.w) + ',' + \
                   str(oi.h) + ',' + \
                   predict['image'] + '\n'
            self.tracksFile.write(line) 
    
    def close(self):
        self.tracksFile.close()
        

