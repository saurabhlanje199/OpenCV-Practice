import cv2
import mediapipe as mp

class FindHands():
    def __init__(self, detection_con=0.5, tracking_con=0.5):
        self.mpHands = mp.solutions.mediapipe.python.solutions.hands
        self.hands = self.mpHands.Hands(min_detection_confidence=detection_con, min_tracking_confidence=tracking_con)
        self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils

    def getPosition(self, img, indexes, hand_no=0, draw=True):
        lst = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) >= hand_no + 1:
                for id, lm in enumerate(results.multi_hand_landmarks[hand_no].landmark):
                    for index in indexes:
                        if id == index:
                            h, w, c = img.shape
                            x, y = int(lm.x * w), int(lm.y * h)
                            lst.append((x, y))
                if draw:
                    self.mpDraw.draw_landmarks(img, results.multi_hand_landmarks[hand_no],
                                               self.mpHands.HAND_CONNECTIONS)
        return lst

    def index_finger_up(self, img, hand_no=0):
        pos = self.getPosition(img, (6, 8), draw=False)
        try:
            if pos[0][1] >= pos[1][1]:
                return True
            elif pos[0][1] < pos[1][1]:
                return False
        except:
            return "NO HAND FOUND"

    def middle_finger_up(self, img, hand_no=0):
        pos = self.getPosition(img, (10, 12), draw=False)
        try:
            if pos[0][1] >= pos[1][1]:
                return True
            elif pos[0][1] < pos[1][1]:
                return False
        except:
            return "NO HAND FOUND"

    def ring_finger_up(self, img, hand_no=0):
        pos = self.getPosition(img, (14, 16), draw=False)
        try:
            if pos[0][1] >= pos[1][1]:
                return True
            elif pos[0][1] < pos[1][1]:
                return False
        except:
            return "NO HAND FOUND"

    def little_finger_up(self, img, hand_no=0):
        pos = self.getPosition(img, (18, 20), draw=False)
        try:
            if pos[0][1] >= pos[1][1]:
                return True
            elif pos[0][1] < pos[1][1]:
                return False
        except:
            return "NO HAND FOUND"

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    hands = FindHands()
    while True:
        succeed, img = cap.read()
        lst = hands.getPosition(img, 8)
        for pt in lst:
            cv2.circle(img, pt, 5, (0, 255, 0), cv2.FILLED)
        cv2.imshow("Image", img)
        if cv2.waitKey(10) == ord("q"):
            break

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList