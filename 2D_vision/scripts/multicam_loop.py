import cv2
import numpy as np
import pyrealsense2 as rs

"""
This file is to use as a reference for how to call each camera by the 
respective serial number as well as which transform file is applicable
for each camera.
"""
W = 848
H = 480

# ----- Camera 1 (end-effector) -----
REALSENSE_INTRINSICS_CAM_1 = "calib/realsense_intrinsics.intr"
REALSENSE_TF_CAM_1 = "calib/realsense_ee.tf"
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device('220222066259')
config_1.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config_1.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

# ----- Camera 2 (static) -----
REALSENSE_INTRINSICS_CAM_2 = "calib/realsense_intrinsics_camera2.intr"
REALSENSE_TF_CAM_2 = "calib/realsense_camera2.tf"
# NOTE: can only see the last 12 corners for calibration
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device('151322066099')
config_2.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config_2.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

# ----- Camera 3 (static) -----
REALSENSE_INTRINSICS_CAM_3 = "calib/realsense_intrinsics_camera2.intr"
REALSENSE_TF_CAM_3 = "calib/realsense_camera2.tf"
# NOTE: can only see the last 16 corners for calibration
pipeline_3 = rs.pipeline()
config_3 = rs.config()
config_3.enable_device('151322069488')
config_3.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config_3.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

# ----- Camera 4 (static) -----
REALSENSE_INTRINSICS_CAM_4 = "calib/realsense_intrinsics_camera2.intr"
REALSENSE_TF_CAM_4 = "calib/realsense_camera2.tf"
# NOTE: can see all 26 corners for calibration
pipeline_4 = rs.pipeline()
config_4 = rs.config()
config_4.enable_device('151322061880')
config_4.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config_4.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

# ----- Camera 5 (static) -----
REALSENSE_INTRINSICS_CAM_5 = "calib/realsense_intrinsics_camera2.intr"
REALSENSE_TF_CAM_5 = "calib/realsense_camera2.tf"
# NOTE: can only see the last 9 corners for calibration
pipeline_5 = rs.pipeline()
config_5 = rs.config()
config_5.enable_device('151322066932')
config_5.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config_5.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

# start streaming
p1 = pipeline_1.start(config_1)
p2 = pipeline_2.start(config_2)
p3 = pipeline_3.start(config_3)
p4 = pipeline_4.start(config_4)
p5 = pipeline_5.start(config_5)

ds1 = p1.get_device().first_depth_sensor().get_depth_scale()
ds2 = p2.get_device().first_depth_sensor().get_depth_scale()
ds3 = p3.get_device().first_depth_sensor().get_depth_scale()
ds4 = p4.get_device().first_depth_sensor().get_depth_scale()
ds5 = p5.get_device().first_depth_sensor().get_depth_scale()
# align stream
aligned_stream_1 = rs.align(rs.stream.color)
aligned_stream_2 = rs.align(rs.stream.color)
aligned_stream_3 = rs.align(rs.stream.color)
aligned_stream_4 = rs.align(rs.stream.color)
aligned_stream_5 = rs.align(rs.stream.color)

# for i in range(650):
i = 0
while(True):
	# Camera 1
	frames_1 = pipeline_1.wait_for_frames()
	frames_1 = aligned_stream_1.process(frames_1)
	color_frame_1 = frames_1.get_color_frame()
	depth_frame_1 = frames_1.get_depth_frame().as_depth_frame()
	color_image_1 = np.asanyarray(color_frame_1.get_data())
	depth_image_1 = np.asanyarray(depth_frame_1.get_data())
	depth_image_1 = depth_image_1/(depth_image_1.max()/255.0)

	# Camera 2
	frames_2 = pipeline_2.wait_for_frames()
	frames_2 = aligned_stream_2.process(frames_2)
	color_frame_2 = frames_2.get_color_frame()
	depth_frame_2 = frames_2.get_depth_frame().as_depth_frame()
	color_image_2 = np.asanyarray(color_frame_2.get_data())
	depth_image_2 = np.asanyarray(depth_frame_2.get_data())
	depth_image_2 = depth_image_2/(depth_image_2.max()/255.0)

	# Camera 3
	frames_3 = pipeline_3.wait_for_frames()
	frames_3 = aligned_stream_3.process(frames_3)
	color_frame_3 = frames_3.get_color_frame()
	depth_frame_3 = frames_3.get_depth_frame().as_depth_frame()
	color_image_3 = np.asanyarray(color_frame_3.get_data())
	depth_image_3 = np.asanyarray(depth_frame_3.get_data())
	depth_image_3 = depth_image_3/(depth_image_3.max()/255.0)

	# Camera 2
	frames_4 = pipeline_4.wait_for_frames()
	frames_4 = aligned_stream_4.process(frames_4)
	color_frame_4 = frames_4.get_color_frame()
	depth_frame_4 = frames_4.get_depth_frame().as_depth_frame()
	color_image_4 = np.asanyarray(color_frame_4.get_data())
	depth_image_4 = np.asanyarray(depth_frame_4.get_data())
	depth_image_4 = depth_image_4/(depth_image_4.max()/255.0)

	# Camera 2
	frames_5 = pipeline_5.wait_for_frames()
	frames_5 = aligned_stream_5.process(frames_5)
	color_frame_5 = frames_5.get_color_frame()
	depth_frame_5 = frames_5.get_depth_frame().as_depth_frame()
	color_image_5 = np.asanyarray(color_frame_5.get_data())
	depth_image_5 = np.asanyarray(depth_frame_5.get_data())
	depth_image_5 = depth_image_5/(depth_image_5.max()/255.0)

	# Show the images
	cv2.imshow("Camera 1", color_image_1)
	cv2.imshow("Camera 2", color_image_2)
	cv2.imshow("Camera 3", color_image_3)
	cv2.imshow("Camera 4", color_image_4)
	cv2.imshow("Camera 5", color_image_5)
	k = cv2.waitKey(1)
# cv2.destroyAllWindows()
	if k==27:
		break
	elif k==ord('s'):
		print("Saving....")
		print(depth_frame_1.get_distance(430,175))
		# Save the images
		cam1_filename = "/home/franka/frankapy/scripts/socket_Imgs/color/" + str(i) + "_cam1.jpg"
		cam2_filename = "/home/franka/frankapy/scripts/socket_Imgs/color/" + str(i) + "_cam2.jpg"
		cam3_filename = "/home/franka/frankapy/scripts/socket_Imgs/color/" + str(i) + "_cam3.jpg"
		cam4_filename = "/home/franka/frankapy/scripts/socket_Imgs/color/" + str(i) + "_cam4.jpg"
		cam5_filename = "/home/franka/frankapy/scripts/socket_Imgs/color/" + str(i) + "_cam5.jpg"

		cam1_filename_d = "/home/franka/frankapy/scripts/socket_Imgs/depth/" + str(i) + "_cam1.jpg"
		cam2_filename_d = "/home/franka/frankapy/scripts/socket_Imgs/depth/" + str(i) + "_cam2.jpg"
		cam3_filename_d = "/home/franka/frankapy/scripts/socket_Imgs/depth/" + str(i) + "_cam3.jpg"
		cam4_filename_d = "/home/franka/frankapy/scripts/socket_Imgs/depth/" + str(i) + "_cam4.jpg"
		cam5_filename_d = "/home/franka/frankapy/scripts/socket_Imgs/depth/" + str(i) + "_cam5.jpg"
		if i == 0:
			print("i = 0")
			cv2.imwrite(cam2_filename, color_image_2)
			cv2.imwrite(cam3_filename, color_image_3)
			cv2.imwrite(cam4_filename, color_image_4)
			cv2.imwrite(cam5_filename, color_image_5)

			cv2.imwrite(cam2_filename_d, depth_image_2)
			cv2.imwrite(cam3_filename_d, depth_image_3)
			cv2.imwrite(cam4_filename_d, depth_image_4)
			cv2.imwrite(cam5_filename_d, depth_image_5)

		cv2.imwrite(cam1_filename, color_image_1)
		cv2.imwrite(cam1_filename_d, depth_image_1)
		# print(np.unique(depth_image_1))
		
		i += 1
cv2.destroyAllWindows()
		