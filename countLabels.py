"""
Created on Sat Sep  7 09:32:22 2019

Python script to count number of labels

@author: Kim Bjerge
"""
import os

if __name__=='__main__':

    image_dic = 'D:/MAVJF/trainInsects/' # Mixture
    #image_dic = 'D:/MAVJF/testInsects/' # Mixture
    
    #mariehone  0
    #honningbi  1
    #stenhumle  2
    #havehumle  3
    #svirrehvid 4
    #svirregul  5    
    #hvips      6
    #flue       7
    
    indexCounts = []
    for i in range(8):
        indexCounts.append(0)
        
    backgrounds = 0
    for filename in os.listdir(image_dic):
        if(filename.endswith('.txt')):
            file = open(image_dic+filename, 'r')
            content = file.read()
            file.close()
            splitted = content.split('\n')
            lines = len(splitted)
            #print('Lines', lines)
            if lines == 1:
                backgrounds = backgrounds + 1
            joined = '';
            for line in range(lines):
                subsplit = splitted[line].split(' ')
                if len(subsplit) == 5: # required: index x y w h
                    index = int(subsplit[0])
                    if index < 8:
                       indexCounts[index] +=1
            
    print(indexCounts, ' total:', sum(indexCounts))
    print("Background images:", backgrounds)
  
 

