# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 19:35:37 2018

@author: Shih Han Wang & Shakti Suman
"""

import cv2
import sys
import numpy as np


def imCluster(imArgs,clusters):
    imageNumber = 1
    for image in imArgs:
        img = cv2.imread(image)
        reshaped = img.reshape((-1, 3))
        resFloat = np.float32(reshaped)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = int(clusters)
        ret, label, center = cv2.kmeans(resFloat, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        flatImg = center[label.flatten()]
        imgRes = flatImg.reshape(img.shape)
        cv2.imshow('Output Image. Press any key to goto next image!', imgRes)
        print("Press any key to go to next image.")
        cv2.imwrite('clusteredImage' + str(imageNumber) + '.jpg', imgRes)
        imageNumber = imageNumber + 1
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Working On Next Image! Please Wait")


imArgs = sys.argv[1:4]
clusters = sys.argv[4]
imCluster(imArgs,clusters)
#q3.py image1.jpg image2.jpg image3.jpg 12
