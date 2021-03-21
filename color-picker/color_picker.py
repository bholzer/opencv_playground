# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

mouseX = 0
mouseY = 0

def handle_click(event, x, y, flags, param):
	global mouseX, mouseY
	if event == cv2.EVENT_MOUSEMOVE:
		mouseX = x
		mouseY = y

if __name__ == "__main__":
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video",
		help="path to the (optional) video file")
	args = vars(ap.parse_args())

	# if a video path was not supplied, grab the reference
	# to the webcam
	if not args.get("video", False):
		vs = VideoStream(src=0).start()

	# otherwise, grab a reference to the video file
	else:
		vs = cv2.VideoCapture(args["video"])

	# allow the camera or video file to warm up
	time.sleep(2.0)

	cv2.namedWindow("image")
	cv2.setMouseCallback("image", handle_click)

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

		(h, w) = frame.shape[:2]
		# print([h, w])

		b, g, r = (frame[mouseY, mouseX])
		hsv = cv2.cvtColor(np.uint8([[[ b,g,r ]]]), cv2.COLOR_BGR2HSV)
		cv2.putText(
			frame,
			f'rgb: ({r}, {g}, {b}) hsv: ({hsv})',
			(10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.35,
			(0, 0, 255),
			1
		)
		print(f'b: {b}, g: {g}, r: {r}')
		cv2.imshow("image", frame)

		# show the frame to our screen and increment the frame counter
		# cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

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