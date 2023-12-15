import cv2
import mediapipe as mp
import time


cam=cv2.VideoCapture(0)
wcam, hcam = 1280, 720
cam.set(3, wcam)
cam.set(4, hcam)

mphands = mp.solutions.hands  # Creating an Instance for the hand package in mediapipe Module
hands = mphands.Hands()  # Hands() require 4 parameter but they are default by nature i.e. (False,2,0.5,0.5)
mpdraw = mp.solutions.drawing_utils  # to draw the lines between the 21 points of a hand

# To calculate FrameRate
ptime, ctime = 0, 0
while True:
    success, img = cam.read()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Since hands object uses only RGB images

    results = hands.process(rgb_img)
    # print(results.multi_hand_landmarks) #To check in the changes of values it gives when a hand shows in front a
    # screen

    if results.multi_hand_landmarks:
        for each_hand in results.multi_hand_landmarks:
            mpdraw.draw_landmarks(img, each_hand,
                                  mphands.HAND_CONNECTIONS)  # This is to draw points i.e landmarks on each hand and
            # HAND_CONNECTION is to draw lines in between
            for id, lm in enumerate(each_hand.landmark):  # To read all the id and landmarks of an ongoing hand
                h, w, c = img.shape  # shape of the image to convert the landmark coordinate ratios to pixels
                cx, cy = int(lm.x * w), int(lm.y * h)  # converting ratios to pixel
                print(id, cx, cy)

                if id == 0:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), 2, cv2.FILLED)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 12, 45), 2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
