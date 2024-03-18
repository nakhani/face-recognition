import numpy as np
import cv2
from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfilename
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import make_classification

def score(filename1,filename2):
    img1=cv2.imread(filename1)
    img2=cv2.imread(filename2)
    # Detect faces in the image
    try:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
    except:
        e=''
    try:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
    except:
        e=''
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    faces1 = faceCascade.detectMultiScale(
        img1,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    faces2 = faceCascade.detectMultiScale(
        img2,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    # mask on sift
    # mask1 = np.zeros(img1.shape[:2], dtype=np.uint8)
    # cv2.rectangle(mask1, (faces1[0][0],faces1[0][1]), (faces1[0][2],faces1[0][3]), (255), thickness = -1)
    # mask2 = np.zeros(img2.shape[:2], dtype=np.uint8)
    # cv2.rectangle(mask2, (faces2[0][0],faces2[0][1]), (faces2[0][2],faces2[0][3]), (255), thickness = -1)
    print(filename2)
    # if faces1.all() and faces2.all():
        # roi1 = img1[faces1[0][0]:faces1[0][2],faces1[0][1]:faces1[0][3]]
        # roi2 = img2[faces2[0][0]:faces2[0][2],faces2[0][1]:faces2[0][3]]
    # else:
        # roi1=img1
        # roi2=img2
    # if not roi1.all() and roi1.all():
        # roi1 = img1
        # roi2 = img2

    
    # Initiate SIFT detector
    sift=cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    # kp1, des1 = sift.detectAndCompute(img1,None)
    # kp2, des2 = sift.detectAndCompute(img2,None)
    
    back=1
    try:
        kp1, des1 = sift.detectAndCompute(img1, None)
    except:  
        kp1, des1 = [0,0]
        back=0
    try:
        kp2, des2 = sift.detectAndCompute(img2, None)
    except:
        kp2, des2 = [0,0]
        back=0
    
    if back==1:
        # Extract best feature with adaboost
        des1_low=''
        try:
            sel1 = VarianceThreshold(threshold=(5))
            des1_low=sel1.fit_transform(des1)
        except ValueError:
            sel1 = []
        des2_low=''
        try:
            sel2 = VarianceThreshold(threshold=(5))
            des2_low=sel2.fit_transform(des2)
        except ValueError:
            sel2 = []
        # X, y = make_classification(n_samples=1000, n_features=4,
                               # n_informative=2, n_redundant=0,
                               # random_state=0, shuffle=False)
        # clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        # clf.fit(X, y)
        # clf.predict([[0, 0, 0, 0]])

        
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1_low,des2_low, k=2)

        # Apply ratio test
        good = []
        max_match=0
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
                a=len(good)
                percent=(a*100)/len(kp2)
                # print("{} % similarity".format(percent))
                if percent >= max_match:
                    max_match=percent
                # if percent >= 5.00:
                    # print('Match Found')
                # if percent < 5.00:
                    # print('Match not Found')

        # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
        # plt.imshow(img3),plt.show()
        return max_match
    else:
        return 0
    
    
def train(filename1,filename2):
    img1=cv2.imread(filename1)
    img2=cv2.imread(filename2)
    # Detect faces in the image
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    faces1 = faceCascade.detectMultiScale(
        img1,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    faces2 = faceCascade.detectMultiScale(
        img2,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    # mask on sift
    # mask1 = np.zeros(img1.shape[:2], dtype=np.uint8)
    # cv2.rectangle(mask1, (faces1[0][0],faces1[0][1]), (faces1[0][2],faces1[0][3]), (255), thickness = -1)
    # mask2 = np.zeros(img2.shape[:2], dtype=np.uint8)
    # cv2.rectangle(mask2, (faces2[0][0],faces2[0][1]), (faces2[0][2],faces2[0][3]), (255), thickness = -1)


    # roi1 = img1[faces1[0][0]:faces1[0][2],faces1[0][1]:faces1[0][3]]
    # roi2 = img2[faces2[0][0]:faces2[0][2],faces2[0][1]:faces2[0][3]]

    # Initiate SIFT detector
    sift=cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    # Extract best feature with adaboost
    des1_low=''
    try:
        sel1 = VarianceThreshold(threshold=(5))
        des1_low=sel1.fit_transform(des1)
    except ValueError:
        sel1 = []
    des2_low=''
    try:
        sel2 = VarianceThreshold(threshold=(5))
        des2_low=sel2.fit_transform(des2)
    except ValueError:
        sel2 = []
    
    all_des=np.append(des1_low,des2_low, axis=0)

    
    return all_des