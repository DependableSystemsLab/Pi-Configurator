from cv2 import cv2
import os
import numpy as np
import imutils
import glob


def plateDetect(images):
    count = 0
    for image in images:
        count = count + 1

        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 90, 90)

        high_thresh, _ = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lowThresh = 0.5*high_thresh

        edges = cv2.Canny(gray, lowThresh, high_thresh)

        cnts, _ = cv2.findContours(
            edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

        plate = None
        for c in cnts:
            perimeter = cv2.arcLength(c, True)
            edges_count = cv2.approxPolyDP(c, 0.02 * perimeter, True)

            if len(edges_count) == 4:
                x, y, w, h = cv2.boundingRect(c)
                if w > h and h < .75*w and h > .25*w:
                    plate = image[y:y+h, x:x+w]
                    cv2.imwrite('./PlateTruth/Plate_' +
                                str(count)+'.png', plate)
                    break
    print(count)


def sortKeyFunc(s):
    return int(os.path.basename(s)[6:-4])


files = glob.glob('./PlateTruth/*.png')
if len(files) > 0:
    for f in files:
        os.remove(f)

images = [cv2.imread(file) for file in sorted(
    glob.glob("./TestDataDown/*.png"), key=sortKeyFunc)]  # Your local image dataset
print(len(images))
plateDetect(images)
