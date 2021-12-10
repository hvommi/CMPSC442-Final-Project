import mediapipe as mp
import numpy as np
import cv2
import time
from distanceFinder import DistanceFinder


class SquidGame:
    
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 160)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # Get actual width of camera
        self.video_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # Get actual height of camera
        self.video_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Initializing constant game stuff
        self.moveThres = 250 # Threshold for movement for red light. If user movement passes this threshold, they lose
        self.goal = 24 # How close to camera the player should be to win in inches
        self.imGreen = np.zeros((int(self.video_height/10), int(self.video_width), 3), np.uint8) # Green light image
        self.imGreen[:] = (0, 255, 0)
        self.imRed = np.zeros((int(self.video_height/10), int(self.video_width), 3), np.uint8) # Red light image
        self.imRed[:] = (0, 0, 255)
        self.redDelay = 0.5

        # Initializing mediapipe stuff
        self.mp_pose = mp.solutions.pose # Stuff for detecting poses
        self.drawing = mp.solutions.drawing_utils # Stuff for drawing poses

        # Set up distance finder
        self.distanceFinder = DistanceFinder(VIDEO_WIDTH= self.video_width, VIDEO_HEIGHT= self.video_height)

    # Sums up the horizontal coordinates of each pose landmark. Used to detect movement during red light.
    def calc_sum(self, landmarkList):
        tsum = 0
        for landmark in landmarkList[11:34]:
            tsum += (landmark.x * 480)
        return tsum
    
    # Checks if player is in frame using pose landmarks
    def checkInFrame(self, res):
        if res.pose_landmarks:
            landmarkList = res.pose_landmarks.landmark
            try:
                # Checks to see if 70% of your shoulders are visible
                if (landmarkList[11].visibility > 0.7) and (landmarkList[12].visibility > 0.7):
                    return True
                else:
                    return False
            except:
                print("Player not in frame")
        return False

    # Finds speed, categorizes it, and returns it for HMM
    def findSpeed(self, prevDistance, currDistance, greenDur):
        slowThres = 2
        speedCategories = [0, 1] # 0 is slow, 1 is fast
        speed = (abs(currDistance - prevDistance) / greenDur)
        if speed > slowThres:
            return speedCategories[0]
        else:
            return speedCategories[1]

    def startGame(self):
        # Initializing variables: start values, etc.
        prevDistance = 0
        greenStartTime = 0
        greenEndTime = 0
        greenDur = 0
        isGreenInit = False
        redStartTime = 0
        redEndTime = 0
        redDur = 3 + self.redDelay
        isRedInit = False
        lose = False
        win = False
        userSum = 0
        tempSum = 0
        inFrame = False
        gameStart = False

        # Initialize window to fit screen
        windowName = "Squid Game"
        cv2.namedWindow(windowName, flags=cv2.WINDOW_NORMAL)
        
        while not inFrame:
            # Checks to see if subject is in frame based on how many landmarks of human body are visible
            # Game will not start unless player is in frame
            success, frm = self.cap.read()
            res = self.distanceFinder.findPose(frm)
            inFrame = self.checkInFrame(res)
            if not inFrame:
                cv2.putText(frm, "Not fully in frame", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 4)
                cv2.imshow(windowName, frm)
            # Initialize values for start of game
            if inFrame:
                prevDistance = self.distanceFinder.findDistance(res)
                gameStart = True
                currWindow = self.imGreen.copy() # Begin with green light visual

        while gameStart:
            # Returns if read is successful and current frame
            success, frm = self.cap.read()
            res = self.distanceFinder.findPose(frm)
            currDistance = self.distanceFinder.findDistance(res)
            self.drawing.draw_landmarks(frm, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            # Once player is in frame, begin game
            if not isGreenInit:
                # Initialize green light
                isGreenInit = True
                currWindow = self.imGreen.copy()
                greenStartTime = time.time()
                greenEndTime = greenStartTime
                # HMM result would do something here
                greenDur = np.random.randint(1, 5) # Random time intervals: anywhere from 1 to 5 seconds

            if (greenEndTime - greenStartTime) <= greenDur:
                # Do stuff during green light
                currWindow = self.imGreen.copy()
                print("Current distance from goal: %.2fft" % ((currDistance - self.goal) / 12))
                greenEndTime = time.time()

            else:
                # Do stuff once green light is over
                if currDistance <= self.goal:
                    # If player is at goal, player wins
                    win = True

                else:
                    currWindow = self.imRed.copy() # Switch to red light visual
                    # Initialize red light
                    if not isRedInit:
                        isRedInit = True
                        redStartTime = time.time()
                        redEndTime = redStartTime
                        userSum = self.calc_sum(res.pose_landmarks.landmark)
                        # Finds speed during previous green light here. Returns 0 or 1, slow or fast
                        speed = self.findSpeed(prevDistance, currDistance, greenDur)
                        
                    # Add delay for red light to detect movement
                    if (redEndTime - redStartTime) <= self.redDelay:
                        redEndTime = time.time()

                    # Detects movement during red light
                    elif (redEndTime - redStartTime) <= redDur:
                        tempSum = self.calc_sum(res.pose_landmarks.landmark)
                        redEndTime = time.time()
                        if abs(tempSum - userSum) > self.moveThres:
                            print("DEAD ", abs(tempSum - userSum))
                            lose = True

                    # Reset red and green light
                    else:
                        isGreenInit = False
                        isRedInit = False

            # Stuff for displaying game
            cv2.putText(currWindow, "Current distance from goal: %.2fft" % ((currDistance - self.goal) / 12),
                (0, int(self.video_height/10) - 30), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)

            mainWin = np.concatenate((frm, currWindow), axis=0)
            cv2.imshow(windowName, mainWin)

            if cv2.waitKey(1) == 27 or lose or win:
                # Quit game if ESC is pressed, or if player has won or lost
                cv2.destroyAllWindows()
                self.cap.release()
                break

        if lose:
            # Do stuff if player lost
            currWindow = self.imRed.copy()
            cv2.putText(currWindow, "You lose!", (0, int(self.video_height/10) - 30), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
        elif win:
            # Do stuff if player won
            currWindow = self.imGreen.copy()
            cv2.putText(currWindow, "You win!", (0, int(self.video_height/10) - 30), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
        if lose or win:
            mainWin = np.concatenate((frm, currWindow), axis=0)
            cv2.imshow(windowName, mainWin)
        
        cv2.waitKey(0)