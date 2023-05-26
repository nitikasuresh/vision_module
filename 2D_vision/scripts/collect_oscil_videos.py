import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
from os.path import join
import threading
from frankapy import FrankaArm


# ----- Camera of choice #3 ----------
fps = 30
W = 848
H = 480
pipeline = rs.pipeline()
config = rs.config()
config.enable_device('151322069488') # Camera3
#config.enable_device('151322066099') # Camera 2
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, fps)
pipeline.start(config)

# ------------------------------------------------------------------- parameters
# {HOME} = frankapy
save_dir = "./oscil_videos"
os.makedirs (save_dir, exist_ok=True)
num_videos = 5

# NOTE:
# Using time.sleep here feels very ineffective; it seems like it will get off when recording many videos.
# how should I deal with record timing?
# (record when robot stopped, sleep otherwise)
rec_duration = 10 # sec
reset_duration = 5+3
gopose_duration = 1
sleep_duration = reset_duration + gopose_duration + 1.80

def record (save_dir, num_videos, duration):
    print (1)
    time.sleep (sleep_duration) # while robot resetting + repositioning
    print (2)
    for i in range (num_videos):

        timestr = "milk_" + time.strftime("%b%d_%H%M%S") + str(f".mp4") # ex) "fluid_Mar06_200655"
        video_dir = join (save_dir, timestr)
        out = cv2.VideoWriter(video_dir, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W,H))

        print(f"Recording video {i+1}")
        start_time = time.time()
        while time.time()-start_time <= duration:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # Convert image to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Write the frame to the video file
            out.write(color_image)
        
        # Release the saved file after specified duration
        out.release()
        print (f"saved: {video_dir}")
        time.sleep (sleep_duration)
    return None

def control_loop (fa, num_videos):
    print ("control1")
    # fa.open_gripper()
    for i in range (num_videos):
        # reset to initial
        print ("reset")
        fa.reset_pose()

        # goto_pose
        pose = fa.get_pose()
        pose.translation -= np.array ([0.0, 0.0, 0.25])
        fa.goto_pose(pose)
        print ("move1")

        # fa.goto_gripper(0.025, force=60.0)
        # pose = fa.get_pose()
        # print ("move2")

        pose.translation += np.array ([0.25, 0.0, 0.0])
        fa.goto_pose (pose, duration=1.0) # default 3
        print ("move3")

        time.sleep(rec_duration+0.5) # stopped time to record the oscillation
        print (4)
    print ("Reset robot after collection")
    fa.reset_pose()
    return None

# Open Gripper in the beginning?
open = False

if __name__ == "__main__":
    print ("Initializing FrankaArm")
    st = time.time()
    fa = FrankaArm()
    print (f"Took {time.time()-st} s to initialize")

    if open:
        fa.open_gripper()
        time.sleep(4)

    # # close gripper around bottle
    print ("goto_gripper")
    st = time.time()
    fa.goto_gripper(0.025, force=90.0) # speed=0.04 m/s default, 0.025m/0.04m/s = 0.625s
    time.sleep(2)
    print (f"Took {time.time()-st} s for goto_gripper")
    


    video_thread = threading.Thread (target=record, args=(save_dir, num_videos, rec_duration))
    robot_thread = threading.Thread (target=control_loop, args=(fa,num_videos))
    video_thread.start()
    robot_thread.start()

# mask dataset 
# for i in range ():
#     diff = mask[i] - mask[-1]
#     diffs.append (diff)

#     if diff < threshold:
#         break