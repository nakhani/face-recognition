# Save Model Using Pickle
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
import similarityScore
import numpy as np
import pandas as pd
import cv2
import os
import sys
import math
import time
from sklearn import svm
import similarityScore
import match

# dictionary of labels
rnoMapping = {}

def datagen():
    """
    Function: datagen 
    
    Input: List of filenames with their absolute paths
    Output: Training data and labels
    
    Description: This function computes SIFT features for each image in the dataset/train folder, thresholds no. of SIFT feature vectors for an image, assigns label to each keypoint vector of the image and returns the final training data and labels matrices used for feeding the SVM in training phase.
    
    """

    Xtrain=[]
    ytrain=[]
    cnt = 0
            
    filePart="dataset/lfw/"
    # opening the text file 
    with open('dataset/pairsDevTrain.txt','r') as file: 
        # reading each line     
        for line in file: 
            nameFile1="" 
            nameFile2="" 
            c=0
            for word in line.split():        
                if c==0:
                    nameFile1=word
                    nameFile2=word
                elif c==1:
                    w=''
                    nu=int(word)
                    if nu<10:
                        w='000'+word
                    if nu>=10 and nu<100:
                        w='00'+word
                    if nu>=100 and nu<1000:
                        w='000'+word
                    labelFile12=nameFile1
                    labelFile1=nameFile1+"_"+w
                    nameFile1=filePart+labelFile12+"/"+nameFile1+"_"+w+".jpg"
                else:
                    w=''
                    nu=int(word)
                    if nu<10:
                        w='000'+word
                    if nu>=10 and nu<100:
                        w='00'+word
                    if nu>=100 and nu<1000:
                        w='000'+word
                    labelFile22=nameFile2
                    labelFile2=nameFile2+"_"+w
                    nameFile2=filePart+labelFile22+"/"+nameFile2+"_"+w+".jpg"
                c +=1
            # Calcute similarity score between two image
            score=similarityScore.score(nameFile1,nameFile2)
            # score=match.main(nameFile1,nameFile2,'sift')
            # print(nameFile1+"...."+nameFile2)
            # construct dictionary for roll no. mapping
            rnoMapping[labelFile1+"-"+labelFile2] = 1
            # rnoMapping[labelFile1+"-"+labelFile2] = cnt
            # cnt += 1
            Xtrain.append([score])
            ytrain.append(rnoMapping[labelFile1+"-"+labelFile2])
    
    with open('dataset/pairsDevTrainNo.txt','r') as file: 
        # reading each line     
        for line in file: 
            nameFile1="" 
            nameFile2="" 
            c=0
            for word in line.split():        
                if c==0:
                    nameFile1=word
                elif c==1:
                    w=''
                    nu=int(word)
                    if nu<10:
                        w='000'+word
                    if nu>=10 and nu<100:
                        w='00'+word
                    if nu>=100 and nu<1000:
                        w='000'+word
                    labelFile12=nameFile1
                    labelFile1=nameFile1+"_"+w
                    nameFile1=filePart+labelFile12+"/"+nameFile1+"_"+w+".jpg"
                elif c==2:
                    nameFile2=word
                else:
                    w=''
                    nu=int(word)
                    if nu<10:
                        w='000'+word
                    if nu>=10 and nu<100:
                        w='00'+word
                    if nu>=100 and nu<1000:
                        w='000'+word
                    labelFile22=nameFile2
                    labelFile2=nameFile2+"_"+w
                    nameFile2=filePart+labelFile22+"/"+nameFile2+"_"+w+".jpg"
                c +=1
            # Calcute similarity score between two image
            score=similarityScore.score(nameFile1,nameFile2)
            # score=match.main(nameFile1,nameFile2,'sift')
            # print(nameFile1+"...."+nameFile2)
            # construct dictionary for roll no. mapping
            rnoMapping[labelFile1+"-"+labelFile2] = 0
            # rnoMapping[labelFile1+"-"+labelFile2] = cnt
            # cnt += 1
            Xtrain.append([score])
            ytrain.append(rnoMapping[labelFile1+"-"+labelFile2])
    
    # return data and label
    return Xtrain, ytrain 
    
def main():
    # call 'datagen' function to get training and testing data & labels
    Xtrain, ytrain = datagen()
    # convert all matrices to numpy array for fast computation
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    # Xtrain = Xtrain.reshape(1, -1)
    # ytrain = ytrain.reshape(1, -1)



    # training phase: SVM , fit model to training data ------------------------------
    clf = svm.SVC(kernel = 'rbf', C = 10, gamma = 0.00001)
    clf.fit(Xtrain, ytrain)


    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

# main()

