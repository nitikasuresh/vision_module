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

	def __init__(self, object_id, object_class):
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
		self.object_id = object_id
		self.object_class = object_class 

		# initialize object position, velocity, rotation, angular velocity all to zero
		self.object_center_point = np.array([0,0,0])
		self.object_velocity = np.array([0,0,0])
		self.object_rotation = np.array([0,0,0,0])
		self.object_ang_velocity = np.array([0,0,0])

		# size, color, type 
		self.size = "0.03,0.03,0.03" #[m]
		self.color = "0,0,255,1"
		self.type = "block"

	def _return_current_position(self):
		return self.object_center_point

	def _return_current_velocity(self):
		return self.object_velocity

	def _return_current_rotation(self):
		return self.object_rotation

	def _return_current_ang_velocity(self):
		return self.object_ang_velocity
	
	def _return_size(self):
		return self.size 

	def _return_color(self):
		return self.color 

	def _return_type(self):
		return self.type

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

		# crop the image and depth information using the object_bounds -- actually mask image to preserve original image size!!!!

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

		# threshold to only list contours above a certain area - after this, should only have 1!!!!
		# print("\nCountour: ", contours)

		# print("\nN Contours: ", len(contours))

		# draw/calculate the centers of objects
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if area > 800:
				# compute the center of the contour
				M = cv2.moments(cnt)
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
				width = int(np.sqrt(area)/8)


				# TODO: Generating samples not robust with no guarantees this isn't larger than the image!!!!
				width = 5
				pixel_pairs = []
				cx_start = cX - width
				cy_start = cY - width
				for i in range(2*width):
					x = cx_start + i
					for j in range(2*width):
						y = cy_start + j
						pixel_pairs.append([x,y])

				object_center_point_in_world, variance = get_object_center_point_in_world_realsense_robust(
					cX,
					cY,
					pixel_pairs,
					depth_image,
					self.realsense_intrinsics,
					self.realsense_to_ee_transform,
					robot_pose)

				# if variance is too high, then ignore z position update
				if variance > 1e-4:
					print("high variance....ignore z update")
					self.object_center_point = np.array([object_center_point_in_world[0], object_center_point_in_world[1], self.object_center_point[2]])
				else:
					self.object_center_point = np.array([object_center_point_in_world[0], object_center_point_in_world[1], object_center_point_in_world[2]])

		# UPDATE:
			# 1) get an array of pixel pairs on the surface of the object 
			# (perhaps contour perimeter & center, perhaps random samples from middle)

			# 2) pass in the depth image

			# 3) have a new function that gets the object point in world that estimates the z point from these multiple samples

			# 4) have a check if there is a large variation in z prediction and if so, estimate z without depth
		return self.object_center_point

	def get_position_apriltag(self, bounds, verts, robot_pose, translation_matrix):
		"""
		Estimate the object position in the robot's frame given the image, depth,
		object bounds and current robot pose based on AprilTag detection.

		Parameters
        ----------
        color_image: opencv image of the workspace
        depth_image: opencv depth image of the workspace
        bounds: numpy array of the bounding box of the detected object
        robot_pose: the pose of the robot end-effector

        Returns
        -------
        object_center_point: the x,y,z coordinate of the center of the object
        	in the robot's coordinate frame
		"""

		# CODE BELOW USING POINT CLOUD TO ESTIMATE X,Y,Z - SUSCEPTIBLE TO DEPTH ERRORS AT CLOSE DISTANCES!!!!

		minx = np.amin(bounds[:,0], axis=0)
		maxx = np.amax(bounds[:,0], axis=0)
		miny = np.amin(bounds[:,1], axis=0)
		maxy = np.amax(bounds[:,1], axis=0)
		
		obj_points = verts[miny:maxy, minx:maxx].reshape(-1,3)

		zs = obj_points[:,2]
		z = np.median(zs)
		xs = obj_points[:,0]
		ys = obj_points[:,1]
		ys = np.delete(ys, np.where((zs < z - 1) | (zs > z + 1))) # take only y for close z to prevent including background

		x_pos = np.median(xs)
		y_pos = np.median(ys)
		z_pos = z

		variance = np.var(zs) # NOTE: variance > 0.15, then z incorrect????

		median_point = np.array([x_pos, y_pos, z_pos])
		object_median_point = get_object_center_point_in_world_realsense_3D_camera_point(median_point, self.realsense_intrinsics, self.realsense_to_ee_transform, robot_pose)

		# if variance > 0.02:
		# 	self.object_center_point = np.array([object_median_point[0], object_median_point[1], self.object_center_point[2]])
		# else:
		# 	self.object_center_point = np.array([object_median_point[0], object_median_point[1], object_median_point[2]])

		com_depth = np.array([object_median_point[0], object_median_point[1], object_median_point[2]])

		com_nodepth = get_object_center_point_in_world_realsense_3D_camera_point(translation_matrix, self.realsense_intrinsics, self.realsense_to_ee_transform, robot_pose)
		com_nodepth = np.array([com_nodepth[0], com_nodepth[1], com_nodepth[2]])
		com_nodepth[2]+=0.03

		# if depth-based prediction is Nan, only use non-depth-based prediction
		if np.isnan(com_depth.any()):
			com_depth = com_nodepth
		# if the prediction difference between depth and no depth is large ignore depth-based z
		elif abs(com_depth[2] - com_nodepth[2]) > 0.1:
			com_depth[2] = com_nodepth[2]

		# scale the no-depth y estimate to account for some linear error we determined experimentally
		delta_y = -0.22*com_depth[2] + 0.11
		com_nodepth[1]-=delta_y

		# weighted average
		self.object_center_point = np.array([(com_depth[0] + com_nodepth[0])/2, (com_depth[1] + com_nodepth[1])/2, (2*com_depth[2] + com_nodepth[2])/3])

		return self.object_center_point

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