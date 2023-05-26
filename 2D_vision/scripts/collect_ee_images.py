import cv2
import numpy as np
import pyrealsense2 as rs

W = 848
H = 480

# ----- Camera 1 (end-effector) -----
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device('220222066259')
config_1.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config_1.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

pipeline_1.start(config_1)
aligned_stream_1 = rs.align(rs.stream.color)

i=0
while True:
    frames_1 = pipeline_1.wait_for_frames()
    frames_1 = aligned_stream_1.process(frames_1)
    color_frame_1 = frames_1.get_color_frame()
    depth_frame_1 = frames_1.get_depth_frame().as_depth_frame()
    color_image_1 = np.asanyarray(color_frame_1.get_data())

    cv2.imshow("Camera 1", color_image_1)
    cv2.waitKey(1)

    # cam1_filename = "scripts/Block_Imgs/" + str(i) + "_cam1.jpg"
    # cv2.imwrite(cam1_filename, color_image_1)
    # i+=1