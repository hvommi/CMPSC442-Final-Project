import cv2
from distanceFinder import DistanceFinder
from ObjectDetectionGame.TestSquidGame import SquidGame


if __name__ == "__main__":
    squidGame = SquidGame()
    squidGame.startGame()

def testDistanceFinder():
    # Camera setup
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
    # Set up distance finder
    distanceFinder = DistanceFinder(VIDEO_WIDTH= video_width, VIDEO_HEIGHT= video_height)
    
    quit = False
    while not quit:
        # Read video capture. First value is if read successfully, second is the current frame
        success, img = cap.read()
        res = distanceFinder.findPose(img)
        # Gets distance in inches
        distance = distanceFinder.findDistance(res)
        distanceFinder.drawDistance(img, res, distance)
        print(distance) # Distance is in inches
        cv2.imshow('Video Feed', img)
        
        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            quit = True
            cap.release()
            cv2.destroyAllWindows()
