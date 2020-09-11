import math
import numpy as np

LEFT = 32
RIGHT = 16
BRANCH_FRAMES = 60

import contextlib

class Trainer:

    def __init__(self, emulator, window, reward, outputfile=None):
        self.emu = emulator
        self.win = window
        self.reward_function_addr = reward
        self.f = outputfile

    def train(self):
        self.emu.savestate.save(9)
        self.emu.input.keypad_add_key(1)
        for i in range(1000):
            self.simulate_branches()

    def simulate_branches(self):
        # slot 9 stores the base savestate
        # slot 8 stores the best branch savestate
        base_save = self.emu.savestate.save(9)

        # the possible branches of the Trainer AI
        branches = [(60, 32), (50, 32), (40, 32),(30, 32), (20, 32), (10, 32), (0, 0),
                    (10, 16), (20, 16), (30, 16),(40, 16), (50, 16), (60, 16)]
        best_branch_reward = -math.inf
        for t, dir in branches:
            frame = 0
            while (frame := frame+1) < BRANCH_FRAMES:
                if frame <= t:
                    self.emu.input.keypad_add_key(dir)
                else:
                    self.emu.input.keypad_rm_key(dir)
                self.emu.cycle()
                self.win.draw()
            branch_reward = self.emu.memory.read(self.reward_function_addr,self.reward_function_addr,2,signed=True)
            branch_pixel = self.emu.screenshot().getpixel((132,294))
            gray_error = np.abs(branch_pixel[0]-branch_pixel[1]) + \
                         np.abs(branch_pixel[0]-branch_pixel[2]) + \
                         np.abs(branch_pixel[2]-branch_pixel[1])
            if gray_error > 40:
                branch_reward = 0
            if self.f:
                print('attempted reward',branch_reward)
                self.f.flush()
            if branch_reward > best_branch_reward:
                best_branch_reward = branch_reward
                self.emu.savestate.save(8)
            self.emu.savestate.load(9)
        self.emu.savestate.load(8)
        self.emu.savestate.save(9)
        if(self.f):
            print('best reward',best_branch_reward,'------\n')
            self.f.flush()

        return best_branch_reward