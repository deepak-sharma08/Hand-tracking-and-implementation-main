import cv2
import mediapipe as mp
import time
import math

class HandDetector:
    def __init__(self, mode=False, maxHand=2, min_det_con=0.5, min_track_con=0.5):
        self.mode, self.maxHand, self.min_det_con, self.min_track_con = mode, maxHand, min_det_con, min_track_con
        self.mphands = mp.solutions.hands  # Creating an Instance for the hand package in mediapipe Module
        self.hands = self.mphands.Hands(self.mode, self.maxHand, self.min_det_con,
                                        self.min_track_con)  # Hands() require 4 parameter but they are default by
        # nature i.e. (False,2,0.5,0.5)
        self.mpdraw = mp.solutions.drawing_utils  # to draw the lines between the 21 points of a hand
        self.finger_tip = [4, 8, 12, 16, 20]

    def Findhands(self, img, draw=True):
        # b_img = cv2.GaussianBlur(img, (9, 9), 5, borderType=cv2.BORDER_CONSTANT)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Since hands object uses only RGB images

        self.results = self.hands.process(
            rgb_img)  # print(results.multi_hand_landmarks) #To check in the changes of values it gives when a hand
        # shows in front a screen

        if self.results.multi_hand_landmarks:
            for each_hand in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, each_hand,
                                               self.mphands.HAND_CONNECTIONS)  # This is to draw points i.e landmarks
                    # on each hand and HAND_CONNECTION is to draw lines in between
        return img

    def FindPosition(self, img, handNo=0, draw=True):

        self.lmlist = []
        self.xlist, self.ylist = [], []
        if self.results.multi_hand_landmarks:
            req_hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(req_hand.landmark):  # To read all the id and landmarks of an ongoing hand
                h, w, c = img.shape  # shape of the image to convert the landmark coordinate ratios to pixels
                cx, cy = int(lm.x * w), int(lm.y * h)  # converting ratios to pixel
                self.xlist.append(cx)
                self.ylist.append(cy)
                self.lmlist.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), 2, cv2.FILLED)

        return self.lmlist

    def drawboundingBox(self, img):
        xmin, xmax = min(self.xlist), max(self.xlist)
        ymin, ymax = min(self.ylist), max(self.ylist)
        bbox=xmin,ymin,xmax,ymax
        cv2.rectangle(img, (xmin-20, ymin-20), (xmax+20, ymax+20), (0, 255, 0), 2)
        return bbox

    def fingersUp(self):
        fingers = []
        if len(self.lmlist) != 0:
            if self.lmlist[self.finger_tip[0]][1] < self.lmlist[self.finger_tip[0] - 1][
                1]:  # Since tip point is 1 points up then the
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range(1, 5):
                if self.lmlist[self.finger_tip[id]][2] < self.lmlist[self.finger_tip[id] - 2][
                    2]:  # Since tip point is 2 points up then the
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers

    def findDistance(self,img,p1,p2,draw=True):
        x1, y1 = self.lmlist[p1][1], self.lmlist[4][2]
        x2, y2 = self.lmlist[p2][1], self.lmlist[p2][2]
        cx1, cy1 = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)  # Circle at point p1
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)  # Circle at point p2
            cv2.circle(img, (cx1, cy1), 10, (0, 0, 255), cv2.FILLED)  # Circle between p1 and p2
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Line between p2 and p2

        length = math.hypot(x2 - x1, y2 - y1)
        return length,img,[x1,y1,x2,y2,cx1,cy1]

def main():
    cam = cv2.VideoCapture(0)
    wcam, hcam = 1280, 720
    cam.set(3, wcam)
    cam.set(4, hcam)
    # To calculate FrameRate
    ptime, ctime = 0, 0
    detector = HandDetector()
    while True:
        success, img = cam.read()
        img = detector.Findhands(img)
        lm_list = detector.FindPosition(img)
        if len(lm_list) != 0:
            print(lm_list[4])
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 12, 45), 2)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


if __name__ == "__main__":
    main()
