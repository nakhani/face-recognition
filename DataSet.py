import os
import math
import random

"""
   Function: LabelData 
   Input: Name of the database, either ORL or Shuffield, note that both 
   Output: Train set and test set
   
   Description: The function retreive the images from each folder, labled the images of the same person with the same label, 
                split the images randomly to train and test set.
   
"""
def LabelData(dataset):
    dataDir = "dataset/"+dataset+"/"
    listDir = os.listdir(dataDir)
    train_set = []
    test_set = []
    classId = 0
    print("---------start data set preprocessing---------------")    
    for individuDir in listDir: 
        
        train_img = os.listdir(dataDir+ individuDir+"/train/")
        test_img = os.listdir(dataDir+ individuDir+"/test/")
        
        for img in train_img:      
            tup = (dataDir+ individuDir+"/train/" + img, individuDir)
            train_set.append(tup)

        
        for img in test_img:
            tup = (dataDir+ individuDir+"/test/" + img, individuDir)
            test_set.append(tup)
            train_set.append(tup)
    print("---------end data set preprocessing---------------")          
    return train_set, test_set 
LabelData("recognition")
