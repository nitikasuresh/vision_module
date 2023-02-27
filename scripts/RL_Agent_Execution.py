import numpy as np
from frankapy import FrankaArm, SensorDataMessageType

from pynput.keyboard import Key, Listener
import time

from DetectObject import DetectObject

# from autolab_core import RigidTransform
# from frankapy import FrankaConstants as FC
# from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
# from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
# from franka_interface_msgs.msg import SensorDataGroup

# import rospy
# import UdpComms as U
import threading
import queue
from pupil_apriltags import Detector

import argparse
# import cv2
import pyrealsense2 as rs
# from cv_bridge import CvBridge
# from autolab_core import RigidTransform, Point

from perception import CameraIntrinsics
from utils import *

import gym
import panda_gym
from panda_gym.envs.panda_tasks import PandaStackEnv
from panda_gym.envs.panda_tasks import PandaPickAndPlaceEnv
from copy import deepcopy as dc
from agent import Agent

REALSENSE_INTRINSICS = "calib/realsense_intrinsics.intr"
REALSENSE_EE_TF = "calib/realsense_ee_shifted.tf"

ALLOW_PRINT = True

def vision_loop(realsense_intrinsics, realsense_to_ee_transform, detected_objects, object_queue):
	# ----- Non-ROS vision attempt
	W = 848
	H = 480

	# Configure depth and color streams
	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_device('220222066259')
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
		detections = detector.detect(bw_image, estimate_tag_pose=True, camera_params=cam_param, tag_size=0.022) ##0028

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


def euler_from_quaternion(quaternion):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quaternion[0]
        y = quaternion[1]
        z = quaternion[2]
        w = quaternion[3]

        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
     
        t2 = np.clip(2.0 * (w * y - z * x), -1, 1)
        pitch_y = np.arcsin(t2)
     
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
     
        return np.array([roll_x, pitch_y, yaw_z])

STACK = True

if __name__ == "__main__":
	## Parameters:
	EPISODE_STEPS = 100 # number of steps per episode.
	RENDER = True # whether to render the agent
	WEIGHT_PATH = 'scripts/agent_weights.pth' #code is run from one folder up. IDK fam.
	COUSTOM_GRIPPER_OFFSET = np.array([-0.06, 0, -0.03])

	# parse inputs to load in camera intrinsics and ee transform.
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS
	)
	parser.add_argument("--extrinsics_file_path", type=str, default=REALSENSE_EE_TF)
	args = parser.parse_args()

	realsense_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
	realsense_to_ee_transform = RigidTransform.load(args.extrinsics_file_path)

	# initialize and reset the fa gripper
	fa = FrankaArm()
	fa.reset_joints()
	pose = fa.get_pose()
	BASE_ROTATION = pose.rotation
	# fa.close_gripper()
	fa.goto_gripper(0.04)

	# Offset for custom finger tips
	COUSTOM_GRIPPER_OFFSET = np.array([-0.03, 0, -0.06])

	goal_rotation = pose.quaternion

	# go to scanning position
	pose.translation = np.array([0.6, 0, 0.5])
	fa.goto_pose(pose)

	# want to create a dictionary that is empty with the unique apriltag ID's 
	detected_objects = {}

	# create a queue to store all of the observations
	object_queue = queue.Queue()

	vision = threading.Thread(target=vision_loop, args=(realsense_intrinsics,realsense_to_ee_transform, detected_objects, object_queue))
	vision.start()

	# Generate the target locations:
	if STACK:
		target_location = np.zeros(3)
		target_location[:2] = np.random.uniform(-0.15, 0.15, 2)
		target1 = np.array([0, 0, 0.02]) + target_location
		target2 = np.array([0, 0, 0.06]) + target_location
	
	else:
		target = np.array([0, 0, 0.02])
		target[0:2] += np.random.uniform(-0.15, 0.15, 2)
		if np.random.random() > 1:
			target[2] += np.random.uniform(0, 0.2)

	# RL Agent Stuff
	if STACK:
		env = PandaStackEnv(render_mode = "human") 
	else:
		env = PandaPickAndPlaceEnv(render_mode = "human")

	# Create the agent
	memory_size = 1e6  # 7e+5 // 50
	batch_size = 256
	actor_lr = 1e-3
	critic_lr = 1e-3
	gamma = 0.98
	tau = 0.05  
	k_future = 0               # Determines what % of the sampled transitions are HER vs ER (k_future = 4 results in 80% HER)
	state_shape = env.observation_space.spaces["observation"].shape 
	n_actions = env.action_space.shape[0]
	n_goals = env.observation_space.spaces["desired_goal"].shape[0]
	action_bounds = [env.action_space.low[0], env.action_space.high[0]]
	agent = Agent(n_states=state_shape,
				n_actions=n_actions,
				n_goals=n_goals,
				action_bounds=action_bounds,
				capacity=memory_size,
				action_size=n_actions,
				batch_size=batch_size,
				actor_lr=actor_lr,
				critic_lr=critic_lr,
				gamma=gamma,
				tau=tau,
				k_future=k_future,
				episode_length = EPISODE_STEPS,
				path="",
				human_file_path = None,
				env=dc(env),
				action_penalty = 0.2)
	agent.load_weights(WEIGHT_PATH)
	agent.set_to_eval_mode()

	POSITION_K = np.array([0.05, 0.05, 0.075])
	GRIPPER_K = 1

	goal_pose = fa.get_pose()
	goal_gripper_width = fa.get_gripper_width()
	gripper_integral = 0

	for _ in range(2):
		start_position = np.random.uniform(-0.15, 0.15, 2)
		start_position_3 = np.zeros(3)
		start_position_3[0:2] = start_position
		while np.linalg.norm(start_position_3 - target1) < 0.1 or np.linalg.norm(start_position_3 - target2) < 0.1:
			start_position = np.random.uniform(-0.15, 0.15, 2)
			start_position_3 = np.zeros(3)
			start_position_3[0:2] = start_position
			start_position_3[2] = 0.02
		
		print('start position is', start_position + np.array([0.6, 0]))

	print('goal position is', target1 + np.array([0.6, 0, 0]), target2 + np.array([0.6, 0, 0]))
	input()
	task_successful = False
	for step in range(50):
		print('step', step)
		current_pose = fa.get_pose()
		current_gripper_width = fa.get_gripper_width()
		current_vel = np.zeros(3)
		current_ang_vel = np.zeros(3)

		# clear out the observation queue, keeping the most recent observation
		# as the detected_objects dictionary.
		queue_size = object_queue.qsize()
		while queue_size > 0:
			detected_objects = object_queue.get()
			queue_size = object_queue.qsize()

		SIM_OFFSET = np.array([-0.6, 0, 0])

		if not STACK:
			if not(0 in detected_objects):
				print('Block not detected')
				continue

			# hardcode the items of interest:
			object = detected_objects[0]
			state = np.concatenate([
				current_pose.position + SIM_OFFSET + COUSTOM_GRIPPER_OFFSET,
				current_vel,
				np.array([current_gripper_width/2]), # put the finger float into an np array
				# np.array([goal_gripper_width/2]),
				object._return_current_position() + SIM_OFFSET + np.array([0, 0, -0.02]),
				euler_from_quaternion(object._return_current_rotation()),
				object._return_current_velocity(),
				object._return_current_ang_velocity(),
			])
			
			if np.linalg.norm(state[0:3] - target) < 0.05:
				task_successful = True
				break

			desired_goal = target
			action = np.array([1, 1, 1, 0.3])*agent.choose_action(state, desired_goal)

			if RENDER:
				env.sim.set_base_pose("object", object._return_current_position() + SIM_OFFSET + np.array([0, 0, -0.02]), np.array([0.0, 0.0, 0.0, 1.0]))
				env.sim.set_base_pose("target", target, np.array([0.0, 0.0, 0.0, 1.0]))

				goal_joints = env.robot.inverse_kinematics(11, current_pose.position + SIM_OFFSET + COUSTOM_GRIPPER_OFFSET, np.array([1, 0, 0, 0]))
				goal_joints[7:9] = current_gripper_width/2
				env.robot.set_joint_angles(goal_joints)

		else:
			if not(0 in detected_objects and 1 in detected_objects):
				print('Blocks not detected')
				continue

			# hardcode the items of interest:
			object1 = detected_objects[0]
			object2 = detected_objects[1]
			state = np.concatenate([
				current_pose.position + SIM_OFFSET + COUSTOM_GRIPPER_OFFSET,
				current_vel,
				# np.array([current_gripper_width/2]), # put the finger float into an np array
				np.array([goal_gripper_width/2]),
				object1._return_current_position() + SIM_OFFSET + np.array([0, 0, -0.02]),
				euler_from_quaternion(object1._return_current_rotation()),
				object1._return_current_velocity(),
				object1._return_current_ang_velocity(),
				object2._return_current_position() + SIM_OFFSET + np.array([0, 0, -0.02]),
				euler_from_quaternion(object2._return_current_rotation()),
				object2._return_current_velocity(),
				object2._return_current_ang_velocity(),
			])
			print(state)

			desired_goal = np.concatenate([target1, target2])
			action = np.array([1, 1, 1, 0.3])*agent.choose_action(state, desired_goal)

			if RENDER:
				env.sim.set_base_pose("object1", object1._return_current_position() + SIM_OFFSET + np.array([0, 0, -0.02]), np.array([0.0, 0.0, 0.0, 1.0]))
				env.sim.set_base_pose("object2", object2._return_current_position() + SIM_OFFSET + np.array([0, 0, -0.02]), np.array([0.0, 0.0, 0.0, 1.0]))
				env.sim.set_base_pose("target1", target1, np.array([0.0, 0.0, 0.0, 1.0]))
				env.sim.set_base_pose("target2", target2, np.array([0.0, 0.0, 0.0, 1.0]))

				goal_joints = env.robot.inverse_kinematics(11, current_pose.position + SIM_OFFSET + COUSTOM_GRIPPER_OFFSET, np.array([1, 0, 0, 0]))
				goal_joints[7:9] = current_gripper_width/2
				env.robot.set_joint_angles(goal_joints)


		
		print('current_state', current_pose.position, current_gripper_width)
		print('action', POSITION_K*action[:3], GRIPPER_K*action[3])
		# input("Press enter to continue")
		position_error = goal_pose.position - current_pose.position
		gripper_error = goal_gripper_width - current_gripper_width

		if (current_gripper_width > 0.001 and current_gripper_width <0.079):
			gripper_integral += 0.1*gripper_error
		
		print('GI:',gripper_integral)

		goal_pose = current_pose
		goal_pose.position += POSITION_K*action[:3]
		goal_pose.rotation = BASE_ROTATION
		goal_gripper_width = np.clip(current_gripper_width + GRIPPER_K*action[3], 0, 0.08)
		print('goal gripper width', goal_gripper_width)

		fa.goto_gripper(np.clip(goal_gripper_width, 0, 0.08))
		print('gripper_done')
		time.sleep(1)
		move_pose = goal_pose
		fa.goto_pose(move_pose)
	
	if task_successful:
		print('Task Successful')
	else:
		print('out of time')

