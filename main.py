import cv2
from distanceFinder import DistanceFinder


webcam = False
cap = cv2.VideoCapture(0)
# Set brightness
cap.set(cv2.CAP_PROP_BRIGHTNESS, 160)
# Set width of camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# Set height of camera
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

distanceFinder = DistanceFinder()
quit = False
while not quit:
    success, img = cap.read()
    distanceFinder.findDistance(img, drawPose=True)
    cv2.imshow('Video Feed', img)
    
    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        quit = True
        cap.release()
        cv2.destroyAllWindows()