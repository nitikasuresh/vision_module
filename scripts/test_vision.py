from frankapy import FrankaArm
from DetectObject import DetectObject
import numpy as np
import argparse
import cv2
import math
import pyrealsense2 as rs
from cv_bridge import CvBridge
from autolab_core import RigidTransform, Point

from perception import CameraIntrinsics
from utils import *

REALSENSE_INTRINSICS = "calib/realsense_intrinsics.intr"
REALSENSE_EE_TF = "calib/realsense_ee.tf"

if __name__ == "__main__":
	"""
	This script uses classical CV methods to find the outlines of the objects in
	the scene as well as their x,y,z coordinates in the robot frame.
	"""

	# load in arguments
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS
	)
	parser.add_argument("--extrinsics_file_path", type=str, default=REALSENSE_EE_TF)
	args = parser.parse_args()
	
	# reset pose and joints
	fa = FrankaArm()
	fa.reset_pose()
	fa.reset_joints()

	# move to middle scanning position
	pose = fa.get_pose()
	pose.translation = np.array([0.6, 0, 0.5])
	fa.goto_pose(pose)

	# begin scanning blocks based on colors
	cv_bridge = CvBridge()
	realsense_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
	realsense_to_ee_transform = RigidTransform.load(args.extrinsics_file_path)

	# create image class
	detection = DetectObject(object_id=0, object_class="block")

	# get a stream of images
	# while True:
	for i in range(20):
		pose = fa.get_pose()
		pose.translation = np.array([0.6, 0, 0.5-i*0.02])
		fa.goto_pose(pose)

		# beginning of loop
		current_pose = fa.get_pose()

		color_image = get_realsense_rgb_image(cv_bridge)
		depth_image = get_realsense_depth_image(cv_bridge)
		object_image_position = np.array([200, 300])

		# meaningless right now - placeholder for updates to the class
		object_bounds = [0,0]

		object_center_point = detection.get_position_image(color_image, depth_image, object_bounds, current_pose)
		print(object_center_point)

		# # image class code begins
		# blur_image = cv2.GaussianBlur(color_image, (5,5),5)

		# # adaptive thresholding on greyscale image
		# gray = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
		# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 6) #25, 6
		# kernal = np.ones((5,5), "uint8")
		# gray_mask = cv2.dilate(thresh, kernal)

		# # create contours
		# contours, hierarchy = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# cont_mask = np.zeros(gray_mask.shape, dtype='uint8')
		# cv2.drawContours(cont_mask, contours, -1, color=(255,255,255), thickness = cv2.FILLED)
		# cv2.imshow("Contours", cont_mask)

		# # threshold to only list contours above a certain area - after this, should only have 1!!!!
		# # print("\nCountour: ", contours)

		# # print("\nN Contours: ", len(contours))

		# # draw/calculate the centers of objects
		# for cnt in contours:
		# 	area = cv2.contourArea(cnt)
		# 	if area > 800:
		# 		# compute the center of the contour
		# 		M = cv2.moments(cnt)
		# 		cX = int(M["m10"] / M["m00"])
		# 		cY = int(M["m01"] / M["m00"])
		# 		width = int(np.sqrt(area)/8)

		# 		pixel_pairs = []
		# 		cx_start = cX - width
		# 		cy_start = cY - width
		# 		for i in range(2*width):
		# 			x = cx_start + i
		# 			for j in range(2*width):
		# 				y = cy_start + j
		# 				pixel_pairs.append([x,y])

		# 		object_center_point_in_world, variance = get_object_center_point_in_world_realsense_robust(
		# 			cX,
		# 			cY,
		# 			pixel_pairs,
		# 			depth_image,
		# 			realsense_intrinsics,
		# 			realsense_to_ee_transform,
		# 			current_pose)

		# 		# if variance is too high, then ignore z position update
		# 		if variance > 1e-4:
		# 			print("high variance....ignore z update")
		# 			object_center_point = np.array([object_center_point_in_world[0], object_center_point_in_world[1], object_center_point[2]])
		# 		else:
		# 			object_center_point = np.array([object_center_point_in_world[0], object_center_point_in_world[1], object_center_point_in_world[2]])


		# 		# image class code ends
		# 		string = "({:0.2f}, {:0.2f}, {:0.2f}) [m]".format(object_center_point[0], object_center_point[1], object_center_point[2])
		# 		area_string = "Area: {:0.2f} [pixel]".format(area)

		# 		# draw contours, COM and area on color image
		# 		cv2.circle(color_image, (cX, cY), 7, (255,255,255), -1)
		# 		cv2.putText(color_image, string, (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		# 		cv2.putText(color_image, area_string, (cX - 35, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		# 		cv2.drawContours(color_image, [cnt], 0, (0,0,255), 2)

		# # Show the images
		# cv2.imshow("Image", color_image)
		# cv2.waitKey(1)