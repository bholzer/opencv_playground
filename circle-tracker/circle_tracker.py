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
import random
import math

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
args = vars(ap.parse_args())

def getCircle(p1, p2, p3):
	xNum = (p1[0]*p1[0]+p1[1]*p1[1])*(p2[1]-p3[1]) + (p2[0]*p2[0]+p2[1]*p2[1])*(p3[1]-p1[1]) + (p3[0]*p3[0]+p3[1]*p3[1])*(p1[1]-p2[1])
	xDenom = ( 2*(p1[0]*(p2[1]-p3[1]) - p1[1]*(p2[0]-p3[0]) + p2[0]*p3[1] - p3[0]*p2[1]) )

	if xDenom == 0:
		return None

	yNum = (p1[0]*p1[0]+p1[1]*p1[1])*(p3[0]-p2[0]) + (p2[0]*p2[0]+p2[1]*p2[1])*(p1[0]-p3[0]) + (p3[0]*p3[0]+p3[1]*p3[1])*(p2[0]-p1[0])
	yDenom = ( 2*(p1[0]*(p2[1]-p3[1]) - p1[1]*(p2[0]-p3[0]) + p2[0]*p3[1] - p3[0]*p2[1]) )

	if yDenom == 0:
		return None


	x = xNum/xDenom
	y = yNum/yDenom
	radius = math.sqrt((x-p1[0])*(x-p1[0]) + (y-p1[1])*(y-p1[1]))
	return [int(x), int(y), int(radius)]

def verifyCircle(x, y, r, edges):
	minInlierDist = 20.0
	maxInlierDistMax = 100.0
	maxInlierDist = r/25.0

	if maxInlierDist < minInlierDist:
		maxInlierDist = minInlierDist
 	
	if maxInlierDist > maxInlierDistMax:
		maxInlierDist = maxInlierDistMax

	C = 2*math.pi
	verifySampleCount = 90

	inlier = 0
	for t in np.arange(0, C, C/verifySampleCount):
		cx = r*math.cos(t) + x
		cy = r*math.sin(t) + y
		if 0 <= cx < edges.shape[1] and 0 <= cy < edges.shape[0] and edges[int(cy)][int(cx)] < maxInlierDist:
			inlier+=1

	return inlier/verifySampleCount

# define the lower and upper boundaries of the "pink"
# ball in the HSV color space
# redLowerBGR = np.uint8([[[0, 0, 200]]])
redLower = (110 , 160, 70)
redUpper = (160, 255, 255)

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

# print(redLowerBGR)

# keep looping
while True:
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
	# blurred = cv2.GaussianBlur(frame, (25, 25), 0)
	blurred = cv2.medianBlur(frame,9)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV) # Intentionally use RGB to swab B&R channels

	# construct a mask for the color, then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, redLower, redUpper)
	mask = cv2.erode(mask, None, iterations=6)
	mask = cv2.dilate(mask, None, iterations=4)

	edges = cv2.Canny(mask,100,200)
	# dilEdge = cv2.dilate(edges, None, iterations=1)
	# inverted = (255 - dilEdge)


	dt = cv2.distanceTransform(255-edges, cv2.DIST_L2, 3)
	points = np.unique(np.argwhere(edges>0), axis=0) # Get coordinates of detected edges

	minRadius = 5
	minPercent = 0.70
	pointCount = points.shape[0]
	bestCircle = [0,0,0,0]

	# RANSAC
	for i in range(pointCount):
		p1, p2, p3 = points[np.random.choice(points.shape[0], 3, replace=False), :]

		if not np.all([p1, p2, p3]):
			continue
		
		circ = getCircle(p1, p2, p3)

		if not circ:
			continue
		
		x,y,r = circ
		# cv2.circle(frame, (y,x), r, (0, 255, 255), 1)
		perc = verifyCircle(y,x,r,dt)

		if perc >= minPercent and r > minRadius and perc > bestCircle[3]:
			bestCircle = [x,y,r,perc]

		if bestCircle[3] >= 0.85:
			break
		
	if bestCircle[3] >= minPercent and bestCircle[2] >= minRadius:
		cv2.circle(frame, (bestCircle[1], bestCircle[0]), bestCircle[2], (0, 255, 255), 1)

	print(bestCircle)

	cv2.imshow("Mask", mask)
	cv2.imshow("Dist", dt)
	cv2.imshow("Frame", frame)

	# show the frame to our screen and increment the frame counter
	# cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	counter += 1

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()


# close all windows
cv2.destroyAllWindows()