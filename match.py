
from __future__ import print_function

import numpy as np
import cv2
from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfilename
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import make_classification
# from common import anorm, getsize

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6


def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv2.SIFT_create()
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv2.SURF_create(800)
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB_create(400)
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv2.BRISK_create()
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher


def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)


def main(fn1,fn2,feature):

    import sys, getopt
    img1 = cv2.imread(fn1, 0)
    img2 = cv2.imread(fn2, 0)
    detector, matcher = init_feature(feature)

    # if img1 is None:
        # print('Failed to load fn1:', fn1)
        # sys.exit(1)
    # if img2 is None:
        # print('Failed to load fn2:', fn2)
        # sys.exit(1)
    # if detector is None:
        # print('unknown feature:', feature_name)
        # sys.exit(1)
    # print('using', feature_name)
    
    # img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    # img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    back=1
    try:
        kp1, desc1 = detector.detectAndCompute(img1, None)
    except:  
        kp1, desc1 = [0,0]
        back=0
    try:
        kp2, desc2 = detector.detectAndCompute(img2, None)
    except:
        kp2, desc2 = [0,0]
        back=0
    
    # Extract best feature with adaboost
    des1_low=''
    try:
        sel1 = VarianceThreshold(threshold=(5))
        des1_low=sel1.fit_transform(desc1)
    except ValueError:
        sel1 = []
    des2_low=''
    try:
        sel2 = VarianceThreshold(threshold=(5))
        des2_low=sel2.fit_transform(desc2)
    except ValueError:
        sel2 = []
    # X, y = make_classification(n_samples=1000, n_features=4,
                           # n_informative=2, n_redundant=0,
                           # random_state=0, shuffle=False)
    # clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    # clf.fit(X, y)
    # clf.predict([[0, 0, 0, 0]])
    # print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))

    def match_and_draw(win):
        # print('matching...')
        raw_matches = matcher.knnMatch(des1_low, trainDescriptors = des2_low, k = 2) #2
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
        if len(p1) >= 4:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            score = (np.sum(status) / len(status) * 100)
            # print('%d%% matched' % score)
        else:
            H, status = None, None
            score =0
            # print('%d matches found, not enough for homography estimation' % len(p1))
        print(score)
        return score
        # _vis = explore_match(win, img1, img2, kp_pairs, status, H)
    if back==1:
        return match_and_draw('find_obj')
    else:
        return 0
        
    # cv2.waitKey()
    # cv2.destroyAllWindows()
