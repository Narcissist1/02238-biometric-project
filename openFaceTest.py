import json
import time
import argparse
import cv2
import itertools
import os
import numpy as np
from random import randint
np.set_printoptions(precision=2)
import openface
from DET_scripts.Compute_and_Plot_DET import compute_and_plot_det, compute_det, plot_det, get_axes_labels

# docker run -p 9000:9000 -p 8000:8000 -v /Users/Mr_ren/DTU/biometrics:/root/biometrics -t -i bamos/openface /bin/bash

modelDir = '/root/openface/models'
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

imgDim = 96
start = time.time()
align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
net = openface.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'), imgDim)

def getRep(imgPath):
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    start = time.time()
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        return None
#         raise Exception("Unable to find a face: {}".format(imgPath))

    start = time.time()
    alignedFace = align.align(imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        return None
#         raise Exception("Unable to align image: {}".format(imgPath))

    start = time.time()
    rep = net.forward(alignedFace)
    return rep


def faceRecognition(img1, img2, threshold=0.4):
    r1 = getRep(img1)
    r2 = getRep(img2)
    if r1 is None or r2 is None:
        return None, None
    d = r1 - r2
    dis = np.dot(d, d)
    if dis < threshold:
        return True, dis
    else:
        return False, dis


def num2str(num):
    num = str(num)
    while len(num) < 3:
        num = '0' + num
    return num


def mated_comparison():
    total = 0
    correct_matched = 0
    false_non_match = 0
    failure_to_acquire = 0
    positives = []
    path = 'data/'
    for i in range(100):
        code = num2str(i) # '000' - '099'
        _path = path + code + '/'
        img1 = _path + code + '_' + str(0) + '.bmp'
        for j in range(1, 5):
            total += 1
            tmp = _path + code + '_' + str(j) + '.bmp'
            rv, dis = faceRecognition(img1, tmp)
            if rv is None:
                failure_to_acquire += 1
                continue
            positives.append(1.0 / dis)
            if rv == True:
                correct_matched += 1
            else:
                false_non_match += 1
    print('Total attempts: ', total)
    print('Accuracy: ', correct_matched)
    print('False non match rate: ', false_non_match)
    print('Failure to acquire rate: ', failure_to_acquire)
    return positives

    
def non_mated_comparison():
    total = 0
    correct_non_match = 0
    false_matched = 0
    failure_to_acquire = 0
    path = 'data/'
    negatives = []
    for i in range(100):
        code = num2str(i) # '000' - '099'    
        _path = path + code + '/'
        for j in range(4):
            total += 1
            img1 = _path + code + '_' + str(randint(0, 4)) + '.bmp'
            k = randint(0, 99)
            while k == i:
                k = randint(0, 99)
            code2 = num2str(k)
            _path2 = path + code2 + '/'
            tmp = _path2 + code2 + '_' + str(randint(0, 4)) + '.bmp'
            rv, dis = faceRecognition(img1, tmp)
            if rv is None:
                failure_to_acquire += 1
                continue
            negatives.append(1.0 / dis)
            if rv == True:
                false_matched += 1
            else:
                correct_non_match += 1
    
    print('Total attempts: ', total)
    print('Accuracy: ', correct_non_match)
    print('False matched rate: ', false_matched)
    print('Failure to acquire rate: ', failure_to_acquire)
    return negatives

if __name__ == '__main__':
    positives = mated_comparison()
    negatives = non_mated_comparison()
    positives = [str(i) for i in positives]
    negatives = [str(i) for i in negatives]
    with open('openface_exp.txt', 'w') as f:
        for item in [positives, negatives]:
            f.write(','.join(item))
            f.write('\n')
