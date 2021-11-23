# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2
def find_marker(image, findContours=1):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	if (cnts):
		if findContours > 1:
			allcnts = []
			cnts = list(cnts)
			cnts.sort(key=cv2.contourArea)
			for c in cnts[-(findContours):]:
				allcnts.append(cv2.minAreaRect(c))
			return allcnts
		c = max(cnts, key = cv2.contourArea)
		# compute the bounding box of the of the paper region and return it
		return [cv2.minAreaRect(c)]
	return []
    
def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 15.0
# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 5.0
# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length

reference_image = cv2.imread("napkin.jpg")
# Image of a 5 in. napkin that is 15 in. away from camera 
marker = find_marker(reference_image)
focalLength = (marker[0][1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

# Finds distance and returns image with distance
def findDistance(img, findContours=1):
    # find the marker in the image, then compute the
	# distance to the marker from the camera
	markers = find_marker(img, findContours)
	if not markers:
		return img
	# draw a bounding box around the image and display it
	for marker in markers:
		# Avoid contours of 0 width
		if marker[1][0] != 0:
			inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
			box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
			box = np.int0(box)
			cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
			cv2.putText(img, "%.2fft" % (inches / 12),
				box[-1], cv2.FONT_HERSHEY_SIMPLEX,
				2.0, (0, 255, 0), 3)
	return img

cv2.imshow("Reference Image", findDistance(reference_image))
cv2.waitKey(0)