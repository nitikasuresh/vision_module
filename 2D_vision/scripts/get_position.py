import argparse
import pyrealsense2 as rs
from utils import *
from frankapy import FrankaArm
from perception import CameraIntrinsics


# dictionary of camera serials
camera_serials = {
    1: '220222066259',
    2: '151322066099',
    3: '151322069488',
    4: '151322061880',
    5: '151322066932'
}

# dictionary of camera intrinsics
camera_intrinsics = {
    1: "calib/realsense_intrinsics.intr",
    2: "calib/realsense_intrinsics_camera2.intr",
    3: "calib/realsense_intrinsics_camera2.intr",
    4: "calib/realsense_intrinsics_camera4.intr",
    5: "calib/realsense_intrinsics_camera5.intr"
}

# dictionary of camera extrinsics
camera_extrinsics = {
    1: "calib/realsense_ee_shifted.tf",
    2: "calib/realsense_camera2.tf",
    3: "calib/realsense_camera3.tf",
    4: "calib/realsense_camera4.tf",
    5: "calib/realsense_camera5.tf"
}

# initialize desired camera
W = 848
H = 480

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_device(camera_serials[1])
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

print("[INFO] start streaming...")
pipeline.start(config)

aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
point_cloud = rs.pointcloud()


# import camera intrinsics and extrinsics
REALSENSE_INTRINSICS = camera_intrinsics[1]
REALSENSE_EE_TF = camera_extrinsics[1]
parser = argparse.ArgumentParser()
parser.add_argument(
    "--intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS
)
parser.add_argument("--extrinsics_file_path", type=str, default=REALSENSE_EE_TF)
args = parser.parse_args()

realsense_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
realsense_to_ee_transform = RigidTransform.load(args.extrinsics_file_path)

# arbitrary pixel coordinate in realsense camera
i, j= 400, 500

# get i, j into x,y,z coordinate in camera frame
frames = pipeline.wait_for_frames()
frames = aligned_stream.process(frames)
color_frame = frames.get_color_frame()
depth_frame = frames.get_depth_frame().as_depth_frame()

points = point_cloud.calculate(depth_frame)
verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, W, 3)

obj_points = verts[i, j].reshape(-1,3)
x = obj_points[:,0] # NOTE: could be an issue here with obj_points.shape --> perhaps need to simply index with obj_points[0]
y = obj_points[:,1] 
z = obj_points[:,2]

# get 3d position in robot frame ---- this code assumes we are using the end-effector camera (stationary cameras do not require a robot pose)
fa = FrankaArm()
robot_pose = fa.get_pose()
com = get_object_center_point_in_world_realsense_3D_camera_point(np.array([x,y,z]), realsense_intrinsics, realsense_to_ee_transform, robot_pose)
# NOTE: use get_object_center_point_in_world_realsense_static_camera() from utils file if using a static camera

# --------- FINAL 3D POINT IN FRANKA WORLD FRAME ----------
com = np.array([com[0], com[1], com[2]]) # should be the x,y,z position in robot frame
# NOTE: now, can call goto_pose to this translation for example
robot_pose.translation = np.array([com[0], com[1], com[2] + 10])
fa.goto_pose(robot_pose)
