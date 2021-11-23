import cv2
from distance import *

refWidth = 256
refHeight = 256
scale = 3


webcam = False
cap = cv2.VideoCapture(0)
# Set brightness
cap.set(cv2.CAP_PROP_BRIGHTNESS, 160)
# Set width of camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# Set height of camera
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# filename = "yep.mp4"
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# framerate = 30
# res = (1920, 1080)
# out = cv2.VideoWriter(filename, fourcc, framerate, res)

quit = False
while not quit:
    success, img = cap.read()
    imgDistance = findDistance(img, findContours=2)
    # out.write(imgDistance)
    cv2.imshow('Video Feed', imgDistance)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        quit = True
        cap.release()
        # out.release()
        cv2.destroyAllWindows()