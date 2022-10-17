import pickle as pkl
import numpy as np

from DetectObject import DetectObject

from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

from frankapy.utils import min_jerk, min_jerk_weight

import rospy
import UdpComms as U
import time
import threading
import queue
import apriltag
from pupil_apriltags import Detector

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

def vision_loop(realsense_intrinsics, realsense_to_ee_transform, detected_objects, object_queue):
	# ----- Non-ROS vision attempt
	W = 848
	H = 480

	# Configure depth and color streams
	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
	config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

	# print("[INFO] start streaming...")
	pipeline.start(config)

	aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
	point_cloud = rs.pointcloud()

	# get a stream of images
	while True:
		# ----- added from other method
		current_pose = fa.get_pose()
		frames = pipeline.wait_for_frames()
		frames = aligned_stream.process(frames)
		color_frame = frames.get_color_frame()
		depth_frame = frames.get_depth_frame().as_depth_frame()

		points = point_cloud.calculate(depth_frame)
		verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, W, 3)  # xyz
		
		# Convert images to numpy arrays
		depth_image = np.asanyarray(depth_frame.get_data())

		# skip empty frames
		if not np.any(depth_image):
			print("no depth")
			# continue

		# print("\n[INFO] found a valid depth frame")
		color_image = np.asanyarray(color_frame.get_data())
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
		detections = detector.detect(bw_image, estimate_tag_pose=True, camera_params=cam_param, tag_size=0.03)
		# print("\nNumber of AprilTags: ", len(detections))

		# loop over the detected AprilTags
		for d in detections:

			# check if apriltag has been detected before
			obj_id = d.tag_id
			# if detected_objects.has_key(obj_id) == False:
			if obj_id not in detected_objects:
				# print("add to dictionary")
				# add tag to the dictionary of detected objects
				tagFamily = d.tag_family.decode("utf-8")
				detected_objects[obj_id] = DetectObject(object_id=obj_id, object_class=tagFamily)

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

			# --------- added code to calculate AprilTag x,y,z position ------
			bounds = np.array([ptA, ptB, ptC, ptD])
			obj = detected_objects[obj_id]
			translation_matrix = d.pose_t
			translation_matrix = np.array(translation_matrix).reshape(3)
			object_center_point = obj.get_position_apriltag(bounds, verts, current_pose, translation_matrix)

			string = "({:0.4f}, {:0.4f}, {:0.4f}) [m]".format(object_center_point[0], object_center_point[1], object_center_point[2])
			cv2.putText(color_image, string, (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

		# put updated dictionary in queue for other thread to access
		object_queue.put(detected_objects)

		# Show the images
		cv2.imshow("Image", color_image)
		cv2.waitKey(1)


def new_object_message(new_object_list, object_dict):
	message = ""
	for new_object in new_object_list:
		new_object = int(new_object)
		message += '_newItem\t' + object_dict[new_object]._return_type() \
		+ '\t' + str(new_object) + '\t' + object_dict[new_object]._return_size() \
		+ '\t' + object_dict[new_object]._return_color() + '\n'
	return message
 
def object_message(object_name, object_dict):
	pos = object_dict[object_name]._return_current_position()
	vel = object_dict[object_name]._return_current_velocity()
	rot = object_dict[object_name]._return_current_rotation()
	avel = object_dict[object_name]._return_current_ang_velocity()
	return str(object_name) + '\t' + str(-pos[1]) + ',' + str(pos[2]) + ',' + str(pos[0]-0.6) + '\t' \
	+ str(-vel[1]) + ',' + str(vel[2]) + ',' + str(vel[0]) + '\t' \
	+ str(rot[1]) + ',' + str(-rot[2]) + ',' + str(-rot[0]) + ',' + str(rot[3]) + '\t' \
	+ str(avel[1]) + ',' + str(-avel[2]) + ',' + str(-avel[0])

# def control(object_queue, ):


if __name__ == "__main__":
	# load in arguments
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS
	)
	parser.add_argument("--extrinsics_file_path", type=str, default=REALSENSE_EE_TF)
	args = parser.parse_args()

	fa = FrankaArm()
	fa.reset_joints()

	pose = fa.get_pose()
	goal_rotation = pose.quaternion
	print("Robot Resting Pose: ", pose)

	print('start socket')
	#change IP
	sock = U.UdpComms(udpIP="172.26.40.95", sendIP = "172.26.90.96", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)
	message_index = 0
	new_object_list = [] # list of all of the objects to render

	i = 0
	dt = 0.02
	rate = rospy.Rate(1 / dt)
	pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=10)
	T = 100
	max_speed = 1 #m/s

	fa = FrankaArm()
	fa.reset_joints()
	pose = fa.get_pose()

	fa.goto_gripper(0, grasp=True)

	# go to scanning position
	pose.translation = np.array([0.6, 0, 0.5])
	fa.goto_pose(pose)

	# begin scanning for blocks

	realsense_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
	realsense_to_ee_transform = RigidTransform.load(args.extrinsics_file_path)

	# want to create a dictionary that is empty with the unique apriltag ID's 
	detected_objects = {}

	# NEED: detected_objects from queue
	object_queue = queue.Queue()

	vision = threading.Thread(target=vision_loop, args=(realsense_intrinsics,realsense_to_ee_transform, detected_objects, object_queue))
	vision.start()
	
	
	initialize = True
	while True:
		hand_position = fa.get_pose().translation
		finger_width = fa.get_gripper_width() # check this 
		message_index += 1

		queue_size = object_queue.qsize()
		while queue_size > 0:
			print("Queue got backed up - removing....")
			detected_objects = object_queue.get()
			queue_size = object_queue.qsize()

		# detected_objects = object_queue.get()

		send_string = str(message_index) + '\n'
		print('detected_objects', detected_objects)
		if len(new_object_list) != 0:
			send_string += new_object_message(new_object_list, detected_objects)
		for game_object in detected_objects:
			send_string += object_message(game_object, detected_objects) + '\n'
		send_string += '_hand\t' + str(-hand_position[1]) + ',' + str(hand_position[2]) + ',' + str(hand_position[0]-0.6) + "," + str(finger_width)
		# new_message = obj_string + '\t0,0,0\t0,0,0,1\t0,0,0\n0,0,0\t0,0,0\t0,0,0,1\t0,0,0\n0,0,0,0'
		sock.SendData(send_string) # Send this string to other application

		# print("New Message: ", new_message)
	
		data = sock.ReadReceivedData() # read data

		# print("Data: ", data)

		if data != None: # if NEW data has been received since last ReadReceivedData function call
			print('send_string', send_string)
			print()
			print(data)
			print('\n')
			unknown_objects, gripper_data = data.split('\n')
			new_object_list = unknown_objects.split('\t')[1:]

			goal_position, goal_width = gripper_data.split('\t')
			cur_pose = fa.get_pose()
			cur_position = cur_pose.translation
			goal_position = np.array(goal_position[1:-1].split(', ')).astype(np.float)
			goal_position = np.array([goal_position[2] + 0.6, -goal_position[0], goal_position[1] + 0.02])
			goal_width = float(goal_width)

			# clip magnitude of goal position to be within max speed bounds
			if not initialize:
				time_diff = timestamp - last_time
				last_time = timestamp
				print("Time Diff:", time_diff)
				if np.linalg.norm(goal_position - cur_position) > max_speed*time_diff:
					goal_position = max_speed*time_diff*(goal_position - cur_position)/np.linalg.norm(goal_position - cur_position) + cur_position

			pose.translation = goal_position
			# # pose.quaternion = goal_rotation

			if initialize:
				# terminate active skills

				fa.goto_pose(pose)
				fa.goto_pose(pose, duration=T, dynamic=True, buffer_time=10,
					cartesian_impedances=[600.0, 600.0, 600.0, 50.0, 50.0, 50.0])
				initialize = False

				init_time = rospy.Time.now().to_time()
				timestamp = rospy.Time.now().to_time() - init_time
				last_time = timestamp
			else:
				timestamp = rospy.Time.now().to_time() - init_time
				traj_gen_proto_msg = PosePositionSensorMessage(
					id=i, timestamp=timestamp,
					position=pose.translation, quaternion=pose.quaternion
				)
				ros_msg = make_sensor_group_msg(
					trajectory_generator_sensor_msg=sensor_proto2ros_msg(
						traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
					)

				rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
				pub.publish(ros_msg)
				rate.sleep()

			i+=1
			rate.sleep()