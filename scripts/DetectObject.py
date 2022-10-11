# import necessary packages and files
from frankapy import FrankaArm
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

# NOTE: Look into python threading!

class DetectObject:
	"""
	NOTE: This camera class is coded for a single wrist-mounted camera!!!! It will
	need to be modified to handle multiple camera inputs!!!!
	"""

	def __init__(self, object_class):
		"""
		"""
		# load in arguments
		parser = argparse.ArgumentParser()
		parser.add_argument(
			"--intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS
		)
		parser.add_argument("--extrinsics_file_path", type=str, default=REALSENSE_EE_TF)
		args = parser.parse_args()

		self.realsense_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
		self.realsense_to_ee_transform = RigidTransform.load(args.extrinsics_file_path)
		self.object_class = object_class 

	def get_position_image(self, color_image, depth_image, object_bounds, robot_pose):
		"""
		Estimate the object position in the robot's frame given the image, depth,
		object bounds and current robot pose based on adaptive thresholding.

		Parameters
        ----------
        color_image: opencv image of the workspace
        depth_image: opencv depth image of the workspace
        object_bounds: numpy array of the bounding box of the detected object
        robot_pose: the pose of the robot end-effector

        Returns
        -------
        object_center_point: the x,y,z coordinate of the center of the object
        	in the robot's coordinate frame
		"""

		# crop the image and depth information using the object_bounds

		# TODO: make the z position robust to depth uncertainty!!!!!
			# store up the depth prediction from multiple points/multiple frames & get the average
			# get the depth prediction from mutliple points on surface and if the variance is too high, scrap it
			# get the depth info and if the difference from previous prediction is too big, then ignore?

		blur_image = cv2.GaussianBlur(color_image, (5,5),5)

		# adaptive thresholding on greyscale image
		gray = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
		thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 6) #25, 6
		kernal = np.ones((5,5), "uint8")
		gray_mask = cv2.dilate(thresh, kernal)

		# create contours
		contours, hierarchy = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cont_mask = np.zeros(gray_mask.shape, dtype='uint8')
		cv2.drawContours(cont_mask, contours, -1, color=(255,255,255), thickness = cv2.FILLED)
		cv2.imshow("Contours", cont_mask)

		print("\nN Contours: ", len(contours))

		# draw/calculate the centers of objects
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if area > 800:
				print("\n\nfinding center...")
				# compute the center of the contour
				M = cv2.moments(cnt)
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])

				object_center_point_in_world = get_object_center_point_in_world_realsense(
					cX,
					cY,
					depth_image,
					self.realsense_intrinsics,
					self.realsense_to_ee_transform,
					robot_pose)

				object_center_point = np.array([object_center_point_in_world[0], object_center_point_in_world[1], object_center_point_in_world[2]])

		return object_center_point

	def get_position_apriltag(self, color_image, depth_image, object_bounds, robot_pose):
		"""
		Estimate the object position in the robot's frame given the image, depth,
		object bounds and current robot pose based on AprilTag detection.

		Parameters
        ----------
        color_image: opencv image of the workspace
        depth_image: opencv depth image of the workspace
        object_bounds: numpy array of the bounding box of the detected object
        robot_pose: the pose of the robot end-effector

        Returns
        -------
        object_center_point: the x,y,z coordinate of the center of the object
        	in the robot's coordinate frame
		"""
		pass 

	def get_color_image(self, color_image, depth_image, object_bounds):
		"""
		Get the color of the object from image input.

		Parameters
        ----------
        color_image: opencv image of the workspace
        depth_image: opencv depth image of the workspace
        object_bounds: numpy array of the bounding box of the detected object

        Returns
        -------
        color: the color of the object (assuming consistent color)
		"""
		pass

	def get_pose_image(self, color_image, depth_image, object_bounds, robot_pose):
		"""
		Estimate the object pose in the robot's frame given the image, depth,
		object bounds and current robot pose based on adaptive thresholding.

		Parameters
        ----------
        color_image: opencv image of the workspace
        depth_image: opencv depth image of the workspace
        object_bounds: numpy array of the bounding box of the detected object
        robot_pose: the pose of the robot end-effector

        Returns
        -------
        object_center_pose: the pose of the center of the object
        	in the robot's coordinate frame
		"""
		pass 

	def get_pose_apriltag(self, color_image, depth_image, object_bounds, robot_pose):
		"""
		Estimate the object pose in the robot's frame given the image, depth,
		object bounds and current robot pose based on AprilTag detection.

		Parameters
        ----------
        color_image: opencv image of the workspace
        depth_image: opencv depth image of the workspace
        object_bounds: numpy array of the bounding box of the detected object
        robot_pose: the pose of the robot end-effector

        Returns
        -------
        object_center_pose: the pose of the center of the object
        	in the robot's coordinate frame
		"""
		pass 

	def get_velocity_image(self, color_image, depth_image, object_bounds, robot_pose):
		"""
		Estimate the object velocity in the robot's frame given the image, depth,
		object bounds and current robot pose based on adaptive thresholding.

		Parameters
        ----------
        color_image: opencv image of the workspace
        depth_image: opencv depth image of the workspace
        object_bounds: numpy array of the bounding box of the detected object
        robot_pose: the pose of the robot end-effector

        Returns
        -------
        object_velocity: the velocity of the center of the object
        	in the robot's coordinate frame
		"""
		pass 

	def get_velocity_apriltag(self, color_image, depth_image, object_bounds, robot_pose):
		"""
		Estimate the object velocity in the robot's frame given the image, depth,
		object bounds and current robot pose based on AprilTag detection.

		Parameters
        ----------
        color_image: opencv image of the workspace
        depth_image: opencv depth image of the workspace
        object_bounds: numpy array of the bounding box of the detected object
        robot_pose: the pose of the robot end-effector

        Returns
        -------
        object_velocity: the velocity of the center of the object
        	in the robot's coordinate frame
		"""
		pass