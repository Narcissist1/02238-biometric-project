import time
import argparse
import cv2
import itertools
import os
import numpy as np
np.set_printoptions(precision=2)
import openface

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
        return None
    d = r1 - r2
    dis = np.dot(d, d)
    if dis < threshold:
        return True
    else:
        return False


def num2str(num):
    num = str(num)
    while len(num) < 3:
        num = '0' + num
    return num


def main():
    total = 0
    correct_matched = 0
    false_non_match = 0
    failure_to_acquire = 0
    path = 'data/'
    for i in range(100):
        code = num2str(i) # '000' - '099'
        _path = path + code + '/'
        img1 = _path + code + '_' + str(0) + '.bmp'
        for j in range(1, 5):
            total += 1
            tmp = _path + code + '_' + str(j) + '.bmp'
            rv = faceRecognition(img1, tmp)
            if rv is None:
                failure_to_acquire += 1
            elif rv == True:
                correct_matched += 1
            else:
                false_non_match += 1
    print('Total attempts: ', total)
    print('Accuracy: ', correct_matched * 1.0 / total)
    print('False non match rate: ', false_non_match * 1.0 / total)
    print('Failure to acquire rate: ', failure_to_acquire * 1.0 / total)


if __name__ == '__main__':
	main()
