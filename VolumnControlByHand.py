import numpy as np
import cv2
import time
import HandTrackingModule
import math
#####################
import pycaw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#####################
wcam, hcam = 1280, 720
####################

cam = cv2.VideoCapture(0)
# am.set(3, wcam)
# cam.set(4, hcam)
ptime = 0
tracker = HandTrackingModule.HandDetector(min_det_con=0.5,maxHand=1)

##################################################################


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
vol_range = volume.GetVolumeRange()

##################################################################

min_vol, max_vol = vol_range[0], vol_range[1]
area = 0
while True:
    img = cam.read()[1]
    img = cv2.flip(img, 1)
    tracker.Findhands(img)
    lmlist = tracker.FindPosition(img, draw=False)
    if len(lmlist) != 0:
        # print(lmlist[4], lmlist[8])
        bbox = tracker.drawboundingBox(img)
        # Filter based on size
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100


        if 50 < area < 1000:
            length, img, lineinfo = tracker.findDistance(img, 4, 8)

            # Convert Volume
            # vol = np.interp(length, [50, 230], [min_vol, max_vol])
            vol_bar = np.interp(length, [50, 230], [400, 150])
            volper = np.interp(length, [50, 230], [0, 100])
            # volume.SetMasterVolumeLevel(vol, None)

            # Reduce resolution to make it smoother
            smooth = 5
            smooth = smooth * round(volper / smooth)
            # Check fingers up
            finger=tracker.fingersUp()
            # Check middle finger up or down
            if finger[2]==0:
                volume.SetMasterVolumeLevelScalar(volper / 100, None)
                cv2.circle(img, (lineinfo[-2], lineinfo[-1]), 10, (0, 255, 0), cv2.FILLED)
            # Drawing
            cvol=int(volume.GetMasterVolumeLevelScalar()*100)
            cv2.putText(img, f'Vol Set: {cvol}', (400,50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)



            cv2.rectangle(img, (50, 150), (70, 400), (0, 255, 0), 3)
            cv2.rectangle(img, (50, int(vol_bar)), (70, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(volper)} %', (40, 438), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 2)
            # cv2.putText(img, '100 %', (50, 130), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 2)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 12, 45), 2)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) == ord('q'):
        break
