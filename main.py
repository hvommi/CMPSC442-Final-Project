import cv2
from distanceFinder import DistanceFinder
from ObjectDetectionGame.SquidGame import runGame


webcam = False
cap = cv2.VideoCapture(0)
# Set brightness
cap.set(cv2.CAP_PROP_BRIGHTNESS, 160)
# Set width of camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# Set height of camera
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# Get width of camera
video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# Get height of camera
video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


distanceFinder = DistanceFinder(VIDEO_WIDTH= video_width, VIDEO_HEIGHT= video_height)
# runGame() #integrate squid game code ^ is line above duplicate?
quit = False
while not quit:
    # Read video capture
    success, img = cap.read()
    distance = distanceFinder.findDistance(img, drawPose=True)
    print(distance) # Distance is in inches
    cv2.imshow('Video Feed', img)
    
    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        quit = True
        cap.release()
        cv2.destroyAllWindows()