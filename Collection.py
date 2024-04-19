# 导入所需的库
import numpy as np
import random
import torch
import utils
import ParaCfg
import Env
import logging
import tqdm
import collections
import graphviz
import sys
import time
import pickle

# ---------------------------- ReplayBuffer ---------------------------
class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 双端队列,先进先出，类似list但更小更快

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))   # ()是创建元组tuple

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions) # 相当于分列打包
        return np.array(state), action, reward, np.array(next_state), done  # 经测试有没有np.array强制转换似乎不影响

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
    
# ---------------------------- Collection ---------------------------
num_episodes = 10

env = Env.Env()

for _ in range(num_episodes):
    DRLstate, EnvInfo = env.reset()
    MctsTree = MCTS(EnvInfo)
    state_list, act_probs_list, V_value_list = [], [], [] # state_list每个element应包含obst，SP/TP 和 【occupied grid】
        
    DoneFlag, expd_cnt = MctsTree.simulate(EnvInfo) # 1:Cnt>expd_maxcnt, 2:openlist = [], 3:PathFnd
    MctsTree.visualize_tree(MctsTree.root_state.MctsNode)
    MctsTree.StoreTreeInfo(MctsTree.root_state.MctsNode, state_list, act_probs_list, V_value_list)
    
    # pickle
    play_data = zip(state_list, act_probs_list, V_value_list)

    if os.path.exists(CONFIG['train_data_buffer_path']):
        while True:
            try:
                with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
                    data_file = pickle.load(data_dict)
                    self.data_buffer = deque(maxlen=self.buffer_size)
                    self.data_buffer.extend(data_file['data_buffer'])
                    self.iters = data_file['iters']
                    del data_file
                    self.iters += 1
                    self.data_buffer.extend(play_data)
                print('成功载入数据')
                break
            except:
                time.sleep(30)
    else:
        self.data_buffer.extend(play_data)
    
    data_dict = {'data_buffer': self.data_buffer, 'iters': self.iters}
    with open(CONFIG['train_data_buffer_path'], 'wb') as data_file:
        pickle.dump(data_dict, data_file)

