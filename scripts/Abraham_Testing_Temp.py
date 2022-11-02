from frankapy import FrankaArm
import numpy as np

fa = FrankaArm()
fa.reset_joints()
fa.close_gripper()
# pose = fa.get_pose()
# theta = np.pi/4
# c = np.cos(theta)
# s = np.sin(theta)
# q = [-0.04962,  0.62539, -0.07684,  0.04657]
# rotation = pose.rotation_from_quaternion(q)
# print(rotation)
# pose.rotation = rotation
# # pose.rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])@np.array([[c,0,s],[0,1,0],[-s,0,c]])
# pose.translation = np.array([0.5, 0, 0.25])
# fa.goto_pose(pose)
# # #pose.rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
# # pose.rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
# # fa.goto_pose(pose)
# # pose.rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# # fa.goto_pose(pose)