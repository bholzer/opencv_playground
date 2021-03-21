# USAGE
# python server.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel --montageW 2 --montageH 2

# import the necessary packages
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
args = vars(ap.parse_args())

# initialize the ImageHub object
imageHub = imagezmq.ImageHub()

frameDict = {}

# initialize the dictionary which will contain  information regarding
# when a device was last active, then store the last time the check
# was made was now
lastActive = {}
lastActiveCheck = datetime.now()

# stores the estimated number of Pis, active checking period, and
# calculates the duration seconds to wait before making a check to
# see if a device was active
ESTIMATED_NUM_PIS = 1
ACTIVE_CHECK_PERIOD = 10
ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD

fps = FPS().start()

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

lower = 5
upper = 60
blur = 3

# start looping over all the frames
while True:
	# receive RPi name and frame from the RPi and acknowledge
	# the receipt
	(rpiName, frame) = imageHub.recv_image()
	imageHub.send_reply(b'OK')

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=800)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (blur, blur), 0)
	edges = cv2.Canny(blurred, lower, upper)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	# cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	# cnts = imutils.grab_contours(cnts)
	# center = None

	# only proceed if at least one contour was found
	# if len(cnts) > 0:
	# 	# find the largest contour in the mask, then use
	# 	# it to compute the minimum enclosing circle and
	# 	# centroid
	# 	c = max(cnts, key=cv2.contourArea)
	# 	M = cv2.moments(c)
	# 	rect = cv2.minAreaRect(c)
	# 	box = np.int0(cv2.boxPoints(rect))
	# 	cv2.drawContours(frame, [box], -0, (0,255,0), 3)

	# if a device is not in the last active dictionary then it means
	# that its a newly connected device
	if rpiName not in lastActive.keys():
		print("[INFO] receiving data from {}...".format(rpiName))

	# record the last active time for the device from which we just
	# received a frame
	lastActive[rpiName] = datetime.now()

	(h, w) = frame.shape[:2]

	# draw the sending device name on the frame
	cv2.putText(frame, rpiName, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	# update the new frame in the frame dictionary
	frameDict[rpiName] = frame

	# if current time *minus* last time when the active device check
	# was made is greater than the threshold set then do a check
	if (datetime.now() - lastActiveCheck).seconds > ACTIVE_CHECK_SECONDS:
		# loop over all previously active devices
		for (rpiName, ts) in list(lastActive.items()):
			# remove the RPi from the last active and frame
			# dictionaries if the device hasn't been active recently
			if (datetime.now() - ts).seconds > ACTIVE_CHECK_SECONDS:
				print("[INFO] lost connection to {}".format(rpiName))
				lastActive.pop(rpiName)
				frameDict.pop(rpiName)

		# set the last active check time as current time
		lastActiveCheck = datetime.now()

	fps.update()
	cv2.imshow("frame", frame)
	cv2.imshow("edges", edges)
	cv2.imshow("blurred", blurred)
	print(f'Upper: {upper}, Lower: {lower}, Blur: {blur}')

	# detect any kepresses
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	if key == ord(","):
		lower -= 5
	if key == ord("."):
		lower += 5
	if key == ord(";"):
		upper -= 5
	if key == ord("'"):
		upper += 5
	if key == ord("["):
		blur -= 2
	if key == ord("]"):
		blur += 2

# do a bit of cleanup
cv2.destroyAllWindows()