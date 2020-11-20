import math
import numpy as np

LEFT = 32
RIGHT = 16
BRANCH_FRAMES = 30

BASE_SAVESTATE = 99

class Teacher:

    def __init__(self, emulator, window, reward, outputfile=None, labelfile=None, headless=True):
        self.emu = emulator
        self.win = window
        self.reward_function_addr = reward
        self.f = outputfile
        self.l = labelfile
        #self.road_colors = ((88,88,72),(120,120,104)) # Vector of colors to represent all the colors on the road
        self.road_colors = ((0,0,0),(255,255,255),(134,77,0),(231,125,36),(231,223,158),(239,69,0),(247,215,28),(77,247,69),(166,247,142),(109,77,12),(142,77,20))
        self.screen_watch_ratio = 0.6 # How to weight the top screen vs bottom screen
        #self.branch_pmf = [1, .0, .0, .0, 0, .00, .00, .00, .00, .00, .00, .00, .00]
        self.branch_pmf = [.85, .0375, .0375, .0375, .0375, .00, .00, .00, .00, .00, .00, .00, .00]
        self.next_dump = self.emu.screenshot() # Hold placeholder data dump
        self.reward_cap = 4100 # The amount of reward before terminating training (i.e. race is done)
        self.headless = headless

        #self.branch_pmf = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        # or [.85, .0375, .0375, .0375, .0375, .00, .00, .00, .00, .00, .00, .00, .00]
        #peach gardens ((192,200,200),(128,96,56),(240,232,176),(216,216,216))
        #shroom ridge ((192,192,192),(112,104,80))
        #the ugly N64 map ((88,88,72),(120,120,104))
        #yoshi falls ((134,77,0),(231,125,36),(231,223,158),(239,69,0),(247,215,28),(77,247,69),(166,247,142),(109,77,12),(142,77,20))

    def train(self):
        self.emu.input.keypad_update(1)
        self.emu.savestate.save(BASE_SAVESTATE)
        delta_progress = []
        for t in range(1000):
            # simulate all the branches at this timestep and get the resulting scores
            pre_progress_score = self.get_reward_memory()
            branch_scores = self.simulate_branches()

            # choose the best score, make it the BASE_STATE for the next timestep t+1, and proceed
            # best_branch = branch_scores[0]
            # Sometimes we intentially choose a suboptimal branch to diversify our training data!
            rank_choose = np.random.choice(np.arange(len(self.branch_pmf)), p=self.branch_pmf)
            next_branch = branch_scores[rank_choose]
            self.emu.savestate.load(next_branch[2])
            self.emu.savestate.save(BASE_SAVESTATE)
            post_progress_score = self.get_reward_memory()

            # Termination conditions: either we win the race or we get stuck
            if self.get_reward_memory() > self.reward_cap:
                break
            if len(delta_progress) > 2 and delta_progress[-1] <= 0 \
                                       and delta_progress[-2] <= 0 \
                                       and delta_progress[-3] <= 0:
                break

            # If we choose the optimal path, create training data
            if rank_choose == 0:
                delta_progress.append(post_progress_score - pre_progress_score)
                if delta_progress[-1] > 0:
                    self.dump_data(branch_scores[0][1])
            else:
                self.l.write('skipped dump\n')
                self.l.flush()

            if self.f:
                print(f'chose {rank_choose} branch')
                print('current progress', self.get_reward_memory())
                print(delta_progress)
                print('chosen reward', next_branch[0], '------\n')
                self.f.flush()

    def dump_data(self, label):
        key = np.datetime64('now').astype(int)
        self.next_dump.save(f'training_data3/{key}.png')
        self.l.write(f'{key}; {label}\n')
        self.l.flush()
        self.next_dump = self.emu.screenshot()

    def color_dist(self, c1, c2):
        return np.min(np.linalg.norm(c1-c2, axis=-1))

    def get_reward_memory(self):
        return self.emu.memory.read(self.reward_function_addr,self.reward_function_addr,2,signed=True)

    def get_current_reward(self):
        reward = self.get_reward_memory()
        pixels_bot = np.asarray(self.emu.screenshot().crop((127, 230, 128, 280)))
        pixels_top = np.asarray(self.emu.screenshot().crop((127, 69, 128, 108)))
        # Scan for pixels in the top screen
        for pix in pixels_top:
            pix = pix[0]
            dist = self.color_dist(pix,self.road_colors)
            if dist > 40:
                reward -= 2*self.screen_watch_ratio
        # Scan for pixels in the bottom screen
        for pix in pixels_bot:
            pix = pix[0]
            dist = self.color_dist(pix,self.road_colors)
            if dist > 40:
                reward -= 2*(1-self.screen_watch_ratio)
        return reward

    def simulate_branch(self, branch):
        t, dir = branch
        frame = 0

        while (frame := frame+1) < BRANCH_FRAMES:
            if frame <= t:
                self.emu.input.keypad_add_key(dir)
            else:
                self.emu.input.keypad_rm_key(dir)
            self.emu.cycle()
            if not self.headless:
                self.win.draw()

        branch_reward = self.get_current_reward()
        if dir == 0:
            # score the branch if it only goes straight, i.e. the 'safe' option
            branch_reward += 3
        if self.f:
            # log results
            #print('attempted reward', branch_reward)
            self.f.flush()

        return branch_reward

    def simulate_branches(self):
        # the possible branches of the Trainer AI
        branches = [(30, 32), (25, 32), (20, 32), (15, 32), (10, 32), (5, 32), (0, 0),
                    (5, 16), (10, 16), (15, 16), (20, 16), (25, 16), (30, 16)]
        branch_scores = []

        for savestate_slot, branch in enumerate(branches):
            # Simulate the branch, log the branch score
            branch_score = self.simulate_branch(branch)
            branch_scores.append((branch_score, branch, savestate_slot))
            self.emu.savestate.save(savestate_slot)
            # Reload the base savestate and retry
            self.emu.savestate.load(BASE_SAVESTATE)

        branch_scores = sorted(branch_scores, key=lambda x: x[0], reverse=True)

        return branch_scores
