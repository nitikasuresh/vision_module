import os
import gym
import panda_gym
from time import sleep
from agent import Agent
from copy import deepcopy as dc

import numpy as np
import time

import UdpComms as U

import threading
import copy
import keyboard

# from panda_gym.envs.panda_tasks.panda_pick_and_place import PandaPickAndPlaceEnv
from panda_gym.envs.panda_tasks.panda_stack import PandaStackEnv

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

def getUnityPosition(py_position):
    # Converts position from python to Unity
    unity_position = np.array([-py_position[1], py_position[2], py_position[0]])
    return unity_position

def getUnityRotation(py_rotation):
    # Converts rotation (and angular velocity) from python to Unity
    if len(py_rotation) == 4:
        unity_rotation = np.array([py_rotation[1], -py_rotation[2], -py_rotation[0], py_rotation[3]])
    elif len(py_rotation) == 3:
        unity_rotation = np.array([py_rotation[1], -py_rotation[2], -py_rotation[0]])
    else:
        raise Exception('Python rotation is not length of 3 or 4. The py_rotation vector is: ' 
            + str(py_rotation) + 'and its length is: ' + str(len(py_rotation)))
    return unity_rotation

def getPythonPosition(unity_position):
    # Converts position from unity to python
    py_position = np.array([unity_position[2], -unity_position[0], unity_position[1]])
    return py_position

def getPythonRotation(unity_rotation):
    # Converts rotation (and angular velocity) from python to Unity
    if len(unity_rotation) == 4:
        py_rotation = np.array([-unity_rotation[2], unity_rotation[0], -unity_rotation[1], unity_rotation[3]])
    elif len(unity_rotation) == 3:
        py_rotation = np.array([-unity_rotation[2], unity_rotation[0], -unity_rotation[1]])
    else:
        raise Exception('Unity rotation is not length of 3 or 4. The unity_rotation vector is: ' + str(unity_rotation) + 'and its length is: ' + str(len(unity_rotation)))
    return py_rotation

class block:
    def __init__(self, position = np.array([0, 0, 0]), rotation = np.array([0, 0, 0, 1]), 
                 velocity = np.array([0, 0, 0]), angular_velocity = np.array([0, 0, 0])):
        self.position = position
        self.rotation = rotation
        self.velocity = velocity
        self.angular_velocity = angular_velocity

    def moveBlock(self, command):
        elements = command.split('\t')
        self.position = getPythonPosition(np.array(elements[1].split(',')).astype(np.float))
        self.velocity = getPythonPosition(np.array(elements[2].split(',')).astype(np.float))
        self.rotation = getPythonRotation(np.array(elements[3].split(',')).astype(np.float))
        self.angular_velocity = getPythonRotation(np.array(elements[4].split(',')).astype(np.float))


class PythonCommunication:
    def __init__(self):
        self.sock = U.UdpComms(udpIP="172.26.76.116", sendIP = "172.26.58.16", portTX=8001, portRX=8000, enableRX=True, suppressWarnings=False)
        self.inventory = {'None':None}
        self.unknown_objects = []
        # pose information is taken in and stored in PandaGym form, then send in Unity form
        self.goal_pose = {"position": None, "rotation": None, "finger":None}
        self.run_send = False
        self.run_recieve = False
        self.last_message = ""
        self.recieved_message = ""
        self.current_pose = {"position": np.zeros(3), 
                             "velocity":np.zeros(3), 
                             "rotation":np.array([0, 0, 0, 1]),
                             "finger":0,
                             "last_time":time.time()}
        
    

    def generateSendString(self):
        position = np.empty(3)
        # clip to virtual walls:
        position[0] = np.clip(comm.goal_pose["position"][0], -0.25, 0.25)
        position[1] = np.clip(comm.goal_pose["position"][1], -0.25, 0.25)
        position[2] = np.clip(comm.goal_pose["position"][2], 0, 0.9)
        finger = np.clip(comm.goal_pose["finger"], 0, 0.0399)


        inventory_message = ""
        for item in self.inventory.keys():
            inventory_message += str(item) + ', '
        inventory_message = inventory_message[:-2] # remove terminal comma and space

        unknown_objects_message = "_unknown"
        for object_name in self.unknown_objects:
            unknown_objects_message += "\t" + object_name

        pose_message = "[" 
        unity_position = getUnityPosition(position)
        for coordinate in unity_position:
            pose_message += str(coordinate) + ", "
        pose_message = pose_message[:-2] #Remove terminal comma and space
        
        unity_rot = getUnityRotation(self.goal_pose["rotation"])
        pose_message += ']\t['
        for coordinate in unity_rot:
            pose_message += str(coordinate) + ", "
        pose_message = pose_message[:-2] # remove terminal comma and space

        pose_message += "]\t"
        pose_message += str(finger)

        return inventory_message + '\n' + unknown_objects_message + '\n' + pose_message


    def parseMessage(self, message):
        commands = message.split('\n')
        message_index = commands[0]
        for command in commands: 
            elements = command.split('\t')
            if elements[0] in {"_deleteItem", "_newItem", "_init"}:
                pass
            elif elements[0] == "_hand":
                self.moveHand(command)
            elif elements[0] in self.inventory:
                self.inventory[elements[0]].moveBlock(command)
            elif elements[0] == "0":
                self.inventory['object1'].moveBlock(command)
            elif elements[0] == "1":
                self.inventory['object2'].moveBlock(command)
            else:
                #add to unknown list
                pass


    def moveHand(self, command):
        elements = command.split("\t")
        new_position = getPythonPosition(np.array(elements[1].split(',')).astype(np.float))
        self.current_pose["rotation"] = getPythonRotation(np.array(elements[2].split(',')).astype(np.float))
        self.current_pose["finger"] = float(elements[3])
        self.current_pose["velocity"] = (new_position - self.current_pose["position"])/(time.time()-self.current_pose["last_time"])
        self.current_pose["last_time"] = time.time()
        self.current_pose["position"] = new_position

    def sendMessage(self):
        while self.run_send:
            time.sleep(0.001)
            send_string = self.generateSendString()
            if send_string != self.last_message:
                self.last_message = send_string
            self.sock.SendData(send_string)
    

    def recieveMessage(self):
        while self.run_recieve:
            time.sleep(0.001)
            data = self.sock.ReadReceivedData()
            if data != None:
                self.recieved_message = data
                self.parseMessage(data)
                # print(data)


    def startCommunication(self, run_revieve=True, run_send=True):
        self.run_recieve = run_revieve
        self.run_send = run_send
        send_thread = threading.Thread(target=comm.sendMessage, daemon=True)
        send_thread.start()
        recieve_thread = threading.Thread(target=comm.recieveMessage, daemon=True)
        recieve_thread.start()


    def stopCommunication(self):
        self.run_recieve = False
        self.run_send = False
        

def init_agent(env, EPISODE_STEPS):
    # directory for storing data
    path = '/Experiments/Play_Data'
    base = os.path.dirname(os.path.realpath(__file__))
    path = base + path
    load_path = os.path.join(path, 'run0')

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
                path=load_path,
                human_file_path = None,
                env=dc(env),
                action_penalty = 0.2)
    return agent

# if __name__ == "__main__":
#     comm = PythonCommunication()
#     comm.unknown_objects = []
#     comm.goal_pose = {"position": [0, 0, 0.3], "rotation": [0, 0, 0, 0], "finger":0}
#     comm.startCommunication()
#     for i in range(100):
#         time.sleep(0.1)
#         comm.goal_pose["finger"] = i/2500
#         print(comm.recieved_message)
#     comm.stopCommunication()

KEYBOARD_CONTROL = False

target_location = np.zeros(3)
target_location[:2] = np.random.uniform(-0.15, 0.15, 2)
comm = PythonCommunication()
comm.inventory = {"object1": block(), 
                  "object2": block(),
                  "target1": block(position=np.array([0, 0, 0.02]) + target_location), 
                  "target2": block(position=np.array([0, 0, 0.06]) + target_location)}
comm.unknown_objects = []
comm.goal_pose = {"position": [0, 0, 0.3], "rotation": [1, 0, 0, 0], "finger":0.02}

position_K = 0.05
gripper_K = 0.2

RENDER = True
EPISODE_STEPS = 50
NUM_SIMS = 20
env = PandaStackEnv(render = RENDER)

weight_path = "C:\\Users\\14127\\Abraham\\Experiments\\Data\\Stack - Humam25p - E200, C20, EP16, ES100, np4, ns0.3, rs61036\\agent_weights.pth"
agent = init_agent(env, EPISODE_STEPS)

# load the agent_weights.pth
agent.load_weights(weight_path)
agent.set_to_eval_mode()

comm.startCommunication(run_send=False)
time.sleep(1)
comm.goal_pose['position'] = comm.current_pose["position"]
comm.goal_pose['rotation'] = comm.current_pose['rotation']
comm.goal_pose['finger'] = comm.current_pose['finger']
comm.stopCommunication()
comm.startCommunication()
print('comm_started')
for n in range(NUM_SIMS):

    for i in range(EPISODE_STEPS):
        state = np.concatenate([
            comm.current_pose["position"],
            comm.current_pose["velocity"],
            np.array([comm.current_pose["finger"]/2]), # put the finger float into an np array
            comm.inventory["object1"].position,
            euler_from_quaternion(comm.inventory["object1"].rotation),
            comm.inventory["object1"].velocity,
            comm.inventory["object1"].angular_velocity,
            comm.inventory["object2"].position,
            euler_from_quaternion(comm.inventory["object2"].rotation),
            comm.inventory["object2"].velocity,
            comm.inventory["object2"].angular_velocity
        ])

        desired_goal = np.concatenate([ 
            comm.inventory['target1'].position,
            comm.inventory['target2'].position])

        action = np.array([1, 1, 1, 0.3])*agent.choose_action(state, desired_goal)
        if RENDER:
            env.sim.set_base_pose("object1", comm.inventory['object1'].position, np.array([0.0, 0.0, 0.0, 1.0]))
            env.sim.set_base_pose("object2", comm.inventory['object2'].position, np.array([0.0, 0.0, 0.0, 1.0]))
            env.sim.set_base_pose("target1", comm.inventory['target1'].position, np.array([0.0, 0.0, 0.0, 1.0]))
            env.sim.set_base_pose("target2", comm.inventory['target2'].position, np.array([0.0, 0.0, 0.0, 1.0]))

            goal_joints = env.robot.inverse_kinematics(11, comm.current_pose["position"], np.array([1, 0, 0, 0]))
            goal_joints[7:9] = comm.current_pose["finger"]/2
            env.robot.set_joint_angles(goal_joints)

        # Step robot:
        if not KEYBOARD_CONTROL:         
            # comm.goal_pose["position"] = np.array([0, 0, 0.3])
            # comm.goal_pose["finger"] = 0
            print('original_pose', comm.goal_pose)
            print('action', position_K*action[:3], gripper_K*action[3])
            if input('Continue?') == "key":
                KEYBOARD_CONTROL = True

            comm.goal_pose["position"] += position_K*action[:3]
            comm.goal_pose["finger"] += gripper_K*action[3]
            comm.goal_pose["finger"] = np.clip(comm.goal_pose["finger"], 0, 0.04)

            print('result_pose', comm.goal_pose)

        # Keyboard control:
        else:
            if keyboard.is_pressed('left'):
                comm.goal_pose["position"] += np.array([0, 0.01, 0])
                print('You Pressed left!')
                print(comm.goal_pose)
            if keyboard.is_pressed('right'):
                comm.goal_pose["position"] += np.array([0, -0.01, 0])
                print('You Pressed right!')
                print(comm.goal_pose)
            if keyboard.is_pressed('down'):
                comm.goal_pose["position"] += np.array([-0.01, 0, 0])
                print('You Pressed down!')
                print(comm.goal_pose)
            if keyboard.is_pressed('up'):
                comm.goal_pose["position"] += np.array([0.01, 0, 0])
                print('You Pressed up!')
                print(comm.goal_pose)
            if keyboard.is_pressed('w'):
                comm.goal_pose["position"] += np.array([0, 0, 0.01])
                print('You Pressed w (up z)!')
                print(comm.goal_pose)
            if keyboard.is_pressed('s'):
                comm.goal_pose["position"] += np.array([0, 0, -0.01])
                print('You Pressed s (down z)!')
                print(comm.goal_pose)
            if keyboard.is_pressed('a') and comm.goal_pose["finger"] < 0.039:
                comm.goal_pose["finger"] += 0.001
                print(comm.goal_pose)
            if keyboard.is_pressed('d') and comm.goal_pose["finger"] > 0.001:
                comm.goal_pose["finger"] -= 0.001
                print(comm.goal_pose)
            # print('updated goal pose')
        sleep(0.1)
        if not KEYBOARD_CONTROL:
            input('sleep')
env.close()
