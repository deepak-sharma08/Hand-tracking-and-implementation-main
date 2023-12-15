import cv2
import numpy
import time
import os

import numpy as np

import HandTrackingModule

cam = cv2.VideoCapture(0)

xp, yp = 0, 0
ptime = 0
detector = HandTrackingModule.HandDetector(min_det_con=0.7,maxHand=1)
folderpath = 'Menu'
myList = os.listdir(folderpath)
overlaylist = []
imgcanvas = np.zeros((720, 1280, 3), np.uint8)
for imgpath in myList:
    image = cv2.imread(f'{folderpath}/{imgpath}')
    overlaylist.append((image))
cam.set(3, 1280)
cam.set(4, 720)
color = (0, 0, 255)
header = overlaylist[0]
brushthickness=5
eraserthickness=50
while True:
    # 1. import the images
    img = cam.read()[1]
    img = cv2.flip(img, 1)

    # 2. Find the Landmarks
    img = detector.Findhands(img)
    lmlist = detector.FindPosition(img, draw=False)

    if len(lmlist) != 0:
        # Tip of the index finger
        x1, y1 = lmlist[8][1:]
        # Tip of the Middle Finger
        x2, y2 = lmlist[12][1:]

    # 3. Check which Finger is up
    finger = detector.fingersUp()

    if len(finger) != 0:
        # 4. Check for selection mode
        if finger[1] == 1 and finger[2] == 1:
            xp,yp=0,0
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlaylist[0]
                    color = (0, 0, 255)
                if 550 < x1 < 750:
                    header = overlaylist[1]
                    color = (0, 255, 0)
                if 800 < x1 < 950:
                    color = (255, 0, 0)
                    header = overlaylist[2]
                if 1000 < x1 < 1200:
                    color = (0, 0, 0)
                    header = overlaylist[3]

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), color, cv2.FILLED)

        # 5. Check for drawing mode
        if finger[1] == 1 and finger[2] == 0 and sum(finger[3:]) == 0:
            cv2.circle(img, (x1, y1), 10, color, cv2.FILLED)
            print('Drawing Mode')
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if color==(0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), color, eraserthickness)
                cv2.line(imgcanvas, (xp, yp), (x1, y1), color, eraserthickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), color, brushthickness)
                cv2.line(imgcanvas, (xp, yp), (x1, y1), color, brushthickness)
            xp, yp = x1, y1

    gray=cv2.cvtColor(imgcanvas,cv2.COLOR_BGR2GRAY)
    inv=cv2.threshold(gray,50,255,cv2.THRESH_BINARY_INV)[1]
    inv=cv2.cvtColor(inv,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,inv)
    img = cv2.bitwise_or(img, imgcanvas)








    img[0:125, 0:1280] = header
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 12, 45), 2)
    #img=cv2.addWeighted(img,0.5,imgcanvas,0.5,0)
    cv2.imshow('Image', img)
    #cv2.imshow('Imagecanvas', imgcanvas)
    if cv2.waitKey(1) == ord('q'):
        break
