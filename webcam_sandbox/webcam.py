# USAGE
# python object_movement.py --video object_tracking_example.mp4
# python object_movement.py

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
args = vars(ap.parse_args())

# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) = (0, 0)
direction = ""

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

def nothing(x):
	pass
	
# Create a black image, a window
frame = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('Frame')


# create trackbars for color change
cv2.createTrackbar('R','Frame',0,255, nothing)
cv2.createTrackbar('G','Frame',0,255, nothing)
cv2.createTrackbar('B','Frame',0,255, nothing)
paused = False

# keep looping
while True:
	key = cv2.waitKey(15) & 0xFF
	counter += 1

	# if the 'q' key is pressed, stop the loop
	if key == ord("p"):
		paused = not paused
	if key == ord("q"):
		break

	if paused:
		time.sleep(2)
		continue
	# grab the current frame
	frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break


	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=800)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	gray = clahe.apply(gray)
	blurred = cv2.GaussianBlur(gray, (3, 3), 0)
	sift = cv2.SIFT_create()
	surf = cv2.xfeatures2d.SURF_create(400)
	kp, des = surf.detectAndCompute(blurred,None)
	keys = cv2.drawKeypoints(blurred, kp, blurred, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	# cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	# cnts = imutils.grab_contours(cnts)
	# center = None

	# only proceed if at least one contour was found
	# if len(cnts) > 0:
	# 	# find the largest contour in the mask, then use
	# 	# it to compute the minimum enclosing circle and
	# 	# centroid
	# 	for c in sorted(cnts, key=lambda x: -cv2.contourArea(x))[:5]:
	# 		M = cv2.moments(c)
	# 		rect = cv2.minAreaRect(c)
	# 		box = np.int0(cv2.boxPoints(rect))
	# 		cv2.drawContours(frame, [box], -0, (0,255,0), 3)

			# center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

	r = cv2.getTrackbarPos('R','Frame')
	g = cv2.getTrackbarPos('G','Frame')
	b = cv2.getTrackbarPos('B','Frame')
	
	cv2.imshow("keys", keys)
	cv2.imshow("gray", blurred)
	cv2.imshow("Frame", frame)


# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()