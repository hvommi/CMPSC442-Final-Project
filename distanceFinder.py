import mediapipe as mp
import numpy as np
import cv2


class DistanceFinder:

	def __init__(self, VIDEO_WIDTH=1920, VIDEO_HEIGHT=1080):
		self.ref_image = cv2.imread("me.jpg")
		# Reference object's distance from camera and width in inches
		self.KNOWN_DISTANCE = 24.0
		self.KNOWN_WIDTH = 14.0

		# Pose detection
		self.mp_pose = mp.solutions.pose
		self.pose = self.mp_pose.Pose()
		self.drawing = mp.solutions.drawing_utils

		# Find focal length based on chest width
		self.processed_pose = self.findPose(self.ref_image)
		self.perWidth = (abs(self.processed_pose.pose_landmarks.landmark[12].x - self.processed_pose.pose_landmarks.landmark[11].x))
		self.focalLength = self.findFocalLength(self.perWidth)
		
		# Resolution of video feed
		self.VIDEO_WIDTH = VIDEO_WIDTH
		self.VIDEO_HEIGHT = VIDEO_HEIGHT
	
	# Finds focal length given perWidth, the percent width of the object 
	def findFocalLength(self, perWidth):
		return (perWidth * self.KNOWN_DISTANCE) / self.KNOWN_WIDTH
	
	# Find distance from person to camera
	def distance_to_camera(self, knownWidth, focalLength, perWidth):
		return (knownWidth * focalLength) / perWidth

	# Finds pose landmarks and returns them. ie pose results
	def findPose(self, img):
		rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		res = self.pose.process(rgb)
		return res

	# Draws pose and distance given the image, pose result, and distance in inches
	def drawDistance(self, img, res, inches):
		self.drawing.draw_landmarks(img, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
		cv2.putText(img, "%.2fft" % (inches / 12),
			(int((res.pose_landmarks.landmark[12].x) * self.VIDEO_WIDTH), int(res.pose_landmarks.landmark[12].y * self.VIDEO_HEIGHT)), cv2.FONT_HERSHEY_SIMPLEX,
			2.0, (0, 255, 0), 3)

	# Finds and returns distance in inches given pose result
	def findDistance(self, res):
		if res.pose_landmarks:
			inches = self.distance_to_camera(self.KNOWN_WIDTH, self.focalLength, abs(res.pose_landmarks.landmark[12].x - res.pose_landmarks.landmark[11].x))
			return inches	
		# If shoulders can't be found, return -1 inches
		else:
			return -1