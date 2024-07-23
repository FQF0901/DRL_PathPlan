"""
@author: Fqf
@time: 20240618
@file: Collection.py
@description: Used to generate training/testing datasets
"""

import collections
import os
import pickle
import Mcts
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import utils

# ==========================================================
# ======================= Collection =======================
# ==========================================================

num_episodes = 3    # need update [important]
DataBuffer = collections.deque(maxlen = 100000)

for _ in range(num_episodes):
    # ---------------------- Simulation ----------------------
    MT = Mcts.MctsTree()
    MT.Simulate()
    MT.VisTree()
    state_list, act_probs_list, V_value_list =  MT.StoreTreeInfo()

    # ------------------------ Pickle ------------------------
    play_data = zip(state_list, act_probs_list, V_value_list)

    if os.path.exists('Mcts_Train_Data_buffer.pkl'):
        try:
            with open('Mcts_Train_Data_buffer.pkl', 'rb') as data_dict:
                data_file = pickle.load(data_dict)
                DataBuffer = collections.deque(maxlen = 100000)   # 每次要清掉，重新压入新数据
                DataBuffer.extend(data_file['DataBuffer'])
                del data_file
                DataBuffer.extend(play_data)
            print('Import data from buffer_pkl success !')
        except:
            print(utils.HighLightRedMsg('Import data from buffer_pkl fail !'))
    else:
        DataBuffer.extend(play_data)
    
    data_dict = {'DataBuffer': DataBuffer}
    with open('Mcts_Train_Data_buffer.pkl', 'wb') as data_file:
        pickle.dump(data_dict, data_file)