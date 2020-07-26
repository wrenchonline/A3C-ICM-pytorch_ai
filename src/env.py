"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import cv2
import numpy as np
import subprocess as sp
from MAMEToolkit.sf_environment import Environment
import os
from MAMEToolkit.sf_environment.Environment import *
from macro import index_to_comb, MACRO_NUMS
import gym

"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import cv2
import numpy as np
import subprocess as sp
from MAMEToolkit.sf_environment import Environment


class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "60", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())


def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (168, 168))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 168, 168))


class StreetFighterEnv(object):
    def __init__(self, index, monitor = None):
        roms_path = "/home/wrench/roms"
        self.env = Environment("env{}".format(index), roms_path)
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None
        self.env.start()

    def step(self, action):
        move_action = action//10
        attack_action = action%10
        frames, reward, round_done, stage_done, game_done = self.env.step(move_action, attack_action)
        if self.monitor:
            for frame in frames:
                self.monitor.record(frame)
        if not (round_done or stage_done or game_done):
            frames = np.concatenate([process_frame(frame) for frame in frames], 0)[None, :, :, :].astype(np.float32)
        else:
            frames = np.zeros((1, 3, 168, 168), dtype=np.float32)
        reward = reward["P1"]
        if stage_done:
            reward = 25
        elif game_done:
            reward = -50
        reward *= (1+(self.env.stage-1)/10)
        reward /= 10
        return frames, reward, round_done, stage_done, game_done

    def reset(self, round_done, stage_done, game_done):
        if game_done:
            self.env.new_game()
        elif stage_done:
            self.env.next_stage()
        elif round_done:
            self.env.next_round()
        return np.zeros((1, 3, 168, 168), dtype=np.float32)
    


class MacroStreetFighterEnv(gym.Env):
    def __init__(self, index, difficulty, monitor=None):
        roms_path = "/home/wrench/roms"
        self.env = Environment("env{}".format(index),
                               roms_path,
                                difficulty=difficulty)
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None
        self.env.start()
        
        self.action_space = gym.spaces.Discrete(18 + MACRO_NUMS)
        self.observation_space = gym.spaces.Box(low=0,
                                                high=1,
                                                shape=(1,3,168,168),
                                                dtype=np.float32)
        #print(self.observation_space)

    def step(self, action):
        frames, reward, round_done, stage_done, game_done = self.step_(action)
        #print(frames)

        if self.monitor:
            for frame in frames:
                self.monitor.record(frame)

        states = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        if not (round_done or stage_done or game_done):
            #states = np.concatenate([process_frame(frame) for frame in frames], 0)[None, :, :, :].astype(np.float32)
            states[:3, :] = process_frame(frames[-1])#搓大招有8 frame ，目前当期异步情况来说严重影响预测
        else:
            states = np.zeros((1, 3, 168, 168), dtype=np.float32)
            #action = 8
            
        #states[:,action + 3, :] = 1
        #print(states.shape) 
        reward = reward["P1"]
        if round_done:
            print("论询完成")
        if stage_done:
            print("游戏小场胜利")
            reward += 25
        elif game_done:
            print("游戏结束")
            reward -= 50

        reward *= (1+(self.env.stage-1)/10)
        reward /= 10

        info = {
            'stage_done': stage_done,
            'round_done': round_done,
            'stage': self.env.stage
        }
        return states, reward, round_done, stage_done, game_done
    #这两个好像是重写 MAMEToolkit Environment 类里面的函数 =-=
    def step_(self, action):
        if self.env.started:
            if not self.env.round_done and not self.env.stage_done and not self.env.game_done:

                if action < 9:
                    #actions = index_to_comb[action - 18]() 
                    actions = index_to_move_action(action)
                elif action < 18:
                    #actions = index_to_comb[action - 18]() 
                    actions = index_to_attack_action(action - 9)
                elif action < 18 + MACRO_NUMS:
                    actions = index_to_comb[action - 18]() #开一个搓大招的接口
                else:
                    actions = []
                    move_action = action//10
                    attack_action = action%10
                    actions += index_to_move_action(move_action)
                    actions += index_to_attack_action(attack_action)
                    #raise EnvironmentError("Action out of range")

                # data = self.env.gather_frames(actions)
                # data = self.check_done(data)
                if action < 18  or  action >= 18 + MACRO_NUMS :
                         data = self.env.gather_frames(actions)
                if  action >= 18    and   action < 18 + MACRO_NUMS:
                        data = self.sub_step_(actions)

                data = self.env.check_done(data)
                return data["frame"], data[
                    "rewards"], self.env.round_done, self.env.stage_done, self.env.game_done
            else:
                raise EnvironmentError(
                    "Attempted to step while characters are not fighting")
        else:
            raise EnvironmentError("Start must be called before stepping")

    def sub_step_(self, actions):
        frames = []
        #搓招居然要8帧
        for step in actions:
            for i in range(step["hold"]):
                data = self.env.emu.step(
                    [action.value for action in step["actions"]])
                frames.append(data['frame'])
        data = self.env.emu.step([])
        frames.append(data['frame'])

        p1_diff = (self.env.expected_health["P1"] - data["healthP1"])
        p2_diff = (self.env.expected_health["P2"] - data["healthP2"])
        self.env.expected_health = {
            "P1": data["healthP1"],
            "P2": data["healthP2"]
        }

        rewards = {"P1": (p2_diff - p1_diff), "P2": (p1_diff - p2_diff)}

        data["rewards"] = rewards
        data["frame"] = frames
        return data

    def reset(self,round_done, stage_done, game_done):
        states = np.zeros(self.observation_space.shape, dtype=np.float32)
        if game_done:
            self.env.new_game()
            states[:3, :] = 1  #重置游戏话目前把他行动改成1-9模式(游戏人物移动)
            #print(states.shape)            
        elif stage_done:
            self.env.next_stage()
        elif round_done:
            self.env.next_round()

        
        return  np.zeros((1, 3, 168, 168), dtype=np.float32)

    def __exit__(self, *args):
        return self.env.close()



def create_train_env(index,difficulty, output_path=None):
    num_inputs = 3
    num_actions = 90
    if output_path:
        monitor = Monitor(384, 224, output_path)
    else:
        monitor = None
    #env = StreetFighterEnv(index, monitor)
    env = MacroStreetFighterEnv(index, difficulty, monitor)

    return env, num_inputs, num_actions
