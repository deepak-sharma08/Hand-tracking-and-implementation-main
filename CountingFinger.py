import cv2
import time
import HandTrackingModule


cam = cv2.VideoCapture(0)
wcam, hcam = 800, 720
cam.set(3, wcam)
cam.set(4, hcam)
ptime = 0

finger_tip = [4, 8, 12, 16, 20]
detector = HandTrackingModule.HandDetector(maxHand=1)
while True:
    img = cam.read()[1]
    img = cv2.flip(img, 1)

    img = detector.Findhands(img)
    lmlist = detector.FindPosition(img, draw=False)

    if len(lmlist) != 0:
        fingers = detector.fingersUp()
        cv2.putText(img, f'{sum(fingers)} Finger', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, f'FPS: {int(fps)}', (580, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Image", img)
    if cv2.waitKey(1) == ord('q'):
        break
