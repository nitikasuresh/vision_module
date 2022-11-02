import cv2
import numpy as np
import pyrealsense2 as rs
# from cv_bridge import CvBridge
from autolab_core import RigidTransform, Point


# 5x8 checkerboard
checkerboard = (5,17) # (5,26) # (5,8) # extended is (5,17)

# lists of points
objpoints = [] # 3d real-world point in robot frame
imgpoints = [] # 2d pixel point in image plane

# ground truth x,y,z position
pose3d = []

# for i in reversed(range(1,checkerboard[1]+1)):
for i in range(1,checkerboard[1]+1):
	# x = 0.1 + i*0.028
	x = 0.34 + i*0.028

	# for j in reversed(range(-2,3)):
	for j in range(-2,3):
		y = j*0.028
		pose3d.append([x,y,0])

pose3d = np.array(pose3d, np.float32)

# print("\nPose 3d: ", pose3d)

# criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Configure depth and color streams
W = 848
H = 480
pipeline = rs.pipeline()
config = rs.config()
config.enable_device('151322061880')
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

pipeline.start(config)
aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
point_cloud = rs.pointcloud()

for i in range(10):
	# get an image from the camera
	frames = pipeline.wait_for_frames()
	frames = aligned_stream.process(frames)
	color_frame = frames.get_color_frame()
	depth_frame = frames.get_depth_frame().as_depth_frame()
	color_image = np.asanyarray(color_frame.get_data())
	bw_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

	# find checkerboard corners
	ret, corners = cv2.findChessboardCorners(bw_image, checkerboard, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

	if ret == True:

		corners_refined = cv2.cornerSubPix(bw_image, corners, (11, 11), (-1, -1), criteria)

		# # visualize the corners in the order they appear
		# for corner in corners:
		# 	print("Corner: ", corner)
		# 	cX = int(corner[0][0])
		# 	cY = int(corner[0][1])
		# 	cv2.circle(color_image, (cX, cY), 5, (0, 0, 255), -1)
		# 	cv2.imshow('image', color_image)
		# 	cv2.waitKey(1000)

		imgpoints.append(np.array(corners_refined.reshape(corners_refined.shape[0],2), np.float32))
		objpoints.append(pose3d)


ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(objpoints, imgpoints, bw_image.shape[::-1], None, None)

print("\nCamera Matrix: ", matrix)

print("\nDistortion: ", distortion)

# print("\nAll Rotation: ", r_vecs)

# print("\nAll Translation: ", t_vecs)

# print("\nRotation Vector Average: ", np.mean(r_vecs, axis=0))

# convert from rotation vector to rotation matrix
rotation_mat, _ = cv2.Rodrigues(np.mean(r_vecs, axis=0))
print("\nRotation Matrix Average: ", rotation_mat)

print("\nTranslation Average: ", np.mean(t_vecs, axis=0))

# swap axis the align with world frame correctly
rvec_mean = np.mean(r_vecs, axis=0)
print("\nAverage rotation vector: ", rvec_mean)
print("\nUpdated Frame Rotation Vector: ", np.array([rvec_mean[2], rvec_mean[0], -rvec_mean[1]]))

new_mat, _ = cv2.Rodrigues(np.array([rvec_mean[2], rvec_mean[0], -rvec_mean[1]]))
print("\nUpdated Rotation Matrix: ", new_mat)

tvec_mean = np.mean(t_vecs, axis=0)
print("\nUpdated Frame Translation: ", np.array([tvec_mean[2], tvec_mean[0], -tvec_mean[1]]))

print("\nInverse Original Rotation: ", np.linalg.inv(rotation_mat))

# print("\nInverse Original Translation: ", np.linalg.inv(tvec_mean))


# TODO: the calibration is wrong, appears to be a swapping of coordinates?
