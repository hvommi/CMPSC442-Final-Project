
import mediapipe as mp
import cv2
import numpy as np
import time
from playsound import playsound

#initializing variables: capture, start values, winner condition, etc.
cap = cv2.VideoCapture(0)
cPos = 0
startT = 0
endT = 0
userSum = 0
dur = 0
isAlive = 1
isInit = False
cStart, cEnd = 0, 0
isCinit = False
tempSum = 0
winner = 0
inFrame = 0
inFramecheck = False
thresh = 180 #tolerated difference between hip and ankle movement

#??? himani will look at this again later.... lol
def calc_sum(landmarkList):
    tsum = 0
    for i in range(11, 33):
        tsum += (landmarkList[i].x * 480)

    return tsum


def calc_dist(landmarkList): #calculates if a left leg moved
    return (landmarkList[28].y * 640 - landmarkList[24].y * 640) #distance between right hip and left ankle

    #442 NOTE - add the other leg??? the arms??

#checks to see if 70% of your legs are visible
def isVisible(landmarkList):
    if (landmarkList[28].visibility > 0.7) and (landmarkList[24].visibility > 0.7):
        return True
    return False

def runGame():
    #draws pose endpoints over image feed using mediapipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    drawing = mp.solutions.drawing_utils

    #red light and green light visual for user to know when to move/not
    im1 = cv2.imread('im1.png')
    im2 = cv2.imread('im2.png')

    currWindow = im1 #begin with green light visual

    while True:

        _, frm = cap.read()
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        frm = cv2.blur(frm, (5, 5))
        drawing.draw_landmarks(frm, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        #442 PROJECT NOTE - ^^^ bring in 'find_markers() code here' - look for a rectangle based on 11+12, 23+24

        if not (inFramecheck):
            #checks to see if subject is in frame based on how many landmarks of human body are visible
            try:
                if isVisible(res.pose_landmarks.landmark):
                    inFrame = 1
                    inFramecheck = True
                else:
                    inFrame = 0
            except:
                print("You are not visible at all")

        if inFrame == 1:
            #once player is in frame, begin game:
            if not (isInit):
                playsound('greenLight.mp3')
                currWindow = im1
                startT = time.time()
                endT = startT
                dur = np.random.randint(1, 5) #random time intervals: anywhere from 1 to 5 seconds
                isInit = True

            if (endT - startT) <= dur:
                try:
                    m = calc_dist(res.pose_landmarks.landmark)
                    if m < thresh:
                        cPos += 1 #if the player's leg hasn't moved (or difference is tolerated), add +1 to progress

                    print("current progress is : ", cPos)
                except:
                    print("Not visible")

                endT = time.time()

            else:
                #winner keeps going until 100 points are accumulated
                if cPos >= 100:
                    print("WINNER")
                    winner = 1

                else:
                    if not (isCinit):
                        isCinit = True
                        cStart = time.time()
                        cEnd = cStart
                        currWindow = im2 #switch to red light visual
                        playsound('redLight.mp3')
                        userSum = calc_sum(res.pose_landmarks.landmark)

                    if (cEnd - cStart) <= 3:
                        tempSum = calc_sum(res.pose_landmarks.landmark)
                        cEnd = time.time()
                        if abs(tempSum - userSum) > 150:
                            print("DEAD ", abs(tempSum - userSum))
                            isAlive = 0

                    else:
                        isInit = False
                        isCinit = False

            cv2.circle(currWindow, ((55 + 6 * cPos), 280), 15, (0, 0, 255), -1)

            mainWin = np.concatenate((cv2.resize(frm, (800, 400)), currWindow), axis=0)
            cv2.imshow("Main Window", mainWin)
        # cv2.imshow("window", frm)
        # cv2.imshow("light", currWindow)

        else:
            cv2.putText(frm, "Please Make sure you are fully in frame", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 4)
            cv2.imshow("window", frm)

        if cv2.waitKey(1) == 27 or isAlive == 0 or winner == 1:
            cv2.destroyAllWindows()
            cap.release()
            break

    frm = cv2.blur(frm, (5, 5))

    if isAlive == 0:
        cv2.putText(frm, "You are Dead", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow("Main Window", frm)

    if winner == 1:
        cv2.putText(frm, "You are Winner", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        cv2.imshow("Main Window", frm)

    cv2.waitKey(0)