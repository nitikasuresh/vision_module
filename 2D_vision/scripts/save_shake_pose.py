import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
from os.path import join
import threading
from frankapy import FrankaArm
from autolab_core import RigidTransform


if __name__ == "__main__":
    fa = FrankaArm()
    pose = fa.get_pose()
    
    fa.open_gripper()
    # pose.translation += np.array ([-0.3, 0.0, 0.1])
    # fa.goto_pose (pose)
    time.sleep (5)

    fa.reset_pose()

    fa.reset_joints()
    
#     #fa.goto_pose (pose)
#     #time.sleep(3)


#     # with np.printoptions(suppress=True):
#     #     print (pose.translation)
#     #     print (pose.rotation)
#     # np.save("./init_trans.npy", pose.translation)
#     # np.save("./init_rot.npy", pose.rotation)
    
#     init_trans = np.load("./init_trans.npy")
#     init_rot = np.load("./init_rot.npy")
#     pose.translation = init_trans
#     pose.rotation = init_rot
#     fa.goto_pose (pose)
#     time.sleep(3)

#     # print (pose)
#     # fa.open_gripper()
#     # time.sleep (3)

#     # reset position manually
#     # pose.translation = np.array ([0.31779758, 0.00125088, 0.48777946])
#     # pose.rotation = np.array ([[ 0.99972715, -0.0190593, -0.01277176],
#     #                            [-0.01936807, -0.99950244, -0.02450541],
#     #                            [-0.01229835,  0.02474609, -0.99961811]])
#     # fa.goto_pose(pose)

#     # #fa.reset_pose()
#     # #pose = fa.get_pose()
#     # print (pose)
#     # time.sleep(2)
    
#     #fa.open_gripper()
#     #time.sleep(2)
#     # print ("RESULT:")
#     # with np.printoptions(suppress=True):
#     #     print (pose.translation)
#     #     print (pose.rotation)
#     # np.save("./shake_trans2.npy", pose.translation)
#     # np.save("./shake_rot2.npy", pose.rotation)
#     # shake2 not completely 90deg


#     #fa.goto_gripper(0.25)
#     #fa.open_gripper()
#     #time.sleep (2)
#     # a = np.load ("./shake_rot.npy")
#     # print (a)
#     #fa.open_gripper()

# ### shaking position
# # RESULT:
# # [0.43653862 0.02414538 0.42738484]
# # [[ 0.99782831 -0.02764921  0.05962375]
# #  [ 0.0590212  -0.02210002 -0.99801203]
# #  [ 0.02891193  0.99936373 -0.02042052]]


# ### New shaking pose today
# # Tra: [ 0.55196535 -0.06157865  0.24191493]
# # Rot: [[ 0.99785815 -0.01902201  0.06243468]
# #  [ 0.06316239  0.04045302 -0.99718301]
# #  [ 0.01644276  0.99899072  0.04156865]]

# ### shake_trans,rot 2
# # RESULT:
# # [0.48517681 0.06922026 0.34414989]
# # [[ 0.99956843 -0.01785893  0.02290801]
# #  [ 0.00284124 -0.72475022 -0.6889985 ]
# #  [ 0.02890736  0.68876623 -0.72440064]]


# ###### NOTE: Reset_pose (unchanged)
# # [ 0.30705774 -0.00010962  0.48677881] trans

# # [[ 0.99999012 -0.00071008 -0.00008247] rot
# #  [-0.00071011 -0.99999004 -0.00041077]
# #  [-0.00008218  0.00041082 -0.99999991]]
