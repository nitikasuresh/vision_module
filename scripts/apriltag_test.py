from frankapy import FrankaArm
import numpy as np
import argparse
import apriltag
from pupil_apriltags import Detector
import cv2
import math
import random
import pyrealsense2 as rs
from cv_bridge import CvBridge
from autolab_core import RigidTransform, Point

from perception import CameraIntrinsics
from utils import *

from autolab_core import RigidTransform, YamlConfig
# from perception_utils.apriltags import AprilTagDetector
# from perception_utils.realsense import get_first_realsense_sensor

REALSENSE_INTRINSICS = "calib/realsense_intrinsics.intr"
REALSENSE_EE_TF = "calib/realsense_ee.tf"

if __name__ == "__main__":
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

	# move to center
	pose = fa.get_pose()
	print("\nRobot Pose: ", pose)
	pose.translation = np.array([0.6, 0, 0.5])
	fa.goto_pose(pose)

	cv_bridge = CvBridge()
	realsense_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
	realsense_to_ee_transform = RigidTransform.load(args.extrinsics_file_path)

	# get a stream of images
	while True:
		color_image = get_realsense_rgb_image(cv_bridge)
		depth_image = get_realsense_depth_image(cv_bridge)

		bw_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

		# python wrapper AprilTag package
		detector = Detector(families="tag36h11",
			nthreads=1,
			quad_decimate=1.0,
			quad_sigma=0.0,
			refine_edges=1,
			decode_sharpening=0.25,
			debug=0)

		# camera parameters [fx, fy, cx, cy]
		cam_param = [realsense_intrinsics.fx, realsense_intrinsics.fy, realsense_intrinsics.cx, realsense_intrinsics.cy]
		# print("Camera Parameters: ", cam_param)
		detections = detector.detect(bw_image, estimate_tag_pose=True, camera_params=cam_param, tag_size=0.03)

		# original AprilTag package
		# detector = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))
		# detections = detector.detect(bw_image)
		print("Number of AprilTags: ", len(detections))

		# loop over the detected AprilTags
		for d in detections:
			# extract the bounding box (x, y)-coordinates for the AprilTag
			# and convert each of the (x, y)-coordinate pairs to integers
			(ptA, ptB, ptC, ptD) = d.corners
			ptB = (int(ptB[0]), int(ptB[1]))
			ptC = (int(ptC[0]), int(ptC[1]))
			ptD = (int(ptD[0]), int(ptD[1]))
			ptA = (int(ptA[0]), int(ptA[1]))

			# draw the bounding box of the AprilTag detection
			cv2.line(color_image, ptA, ptB, (0, 255, 0), 2)
			cv2.line(color_image, ptB, ptC, (0, 255, 0), 2)
			cv2.line(color_image, ptC, ptD, (0, 255, 0), 2)
			cv2.line(color_image, ptD, ptA, (0, 255, 0), 2)

			# draw the center (x, y)-coordinates of the AprilTag
			(cX, cY) = (int(d.center[0]), int(d.center[1]))
			cv2.circle(color_image, (cX, cY), 5, (0, 0, 255), -1)

			# draw the tag family on the image
			tagFamily = d.tag_family.decode("utf-8")
			# cv2.putText(color_image, tagFamily, (ptA[0], ptA[1] - 15),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			# print("[INFO] tag family: {}".format(tagFamily))

			# estimate pose
			print("Rotation Pose: ", d.pose_R)
			print("Translation Pose: ", d.pose_t)
			print("Pose Error: ", d.pose_err)

			# visualize orientation
			# imgpts, jac = cv2.projectPoints(d.corners, d.pose_R, d.pose_t, camera_mat, dist_mat)


			# TODO:
				# 1) convert rotation matrix w.r.t. camera frame to quaternion
				# 2) translate quaternion from camera frame to franka_tool frame
				# 3) given franka_tool to world quaternion translate the object orientation to world frame
				# 4) convert quaternion to axis-angle format for interpretability
				# 5) visualize the orientation in the image livestream


		# Show the images
		cv2.imshow("Image", color_image)
		cv2.waitKey(1)

