# 导入所需的库
import Env
import collections
import pickle
import MCTS
import os
    
# ---------------------------- Collection ---------------------------
num_episodes = 10

env = Env.Env()
DataBuffer = collections.deque(maxlen = 100000)

for _ in range(num_episodes):
    DRLstate, EnvInfo = env.reset()
    MctsTree = MCTS.MCTS(EnvInfo)
    state_list, act_probs_list, V_value_list = [], [], [] # state_list每个element应包含obst，SP/TP 和 【occupied grid】
        
    DoneFlag, expd_cnt = MctsTree.simulate(EnvInfo) # 1:Cnt>expd_maxcnt, 2:openlist = [], 3:PathFnd
    MctsTree.visualize_tree(MctsTree.root_state.MctsNode)
    MctsTree.StoreTreeInfo(MctsTree.root_state.MctsNode, EnvInfo.State, state_list, act_probs_list, V_value_list)
    
    # ---------------------------- pickle ---------------------------
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
            print('Import data from buffer_pkl fail !')
    else:
        DataBuffer.extend(play_data)
    
    data_dict = {'DataBuffer': DataBuffer}
    with open('Mcts_Train_Data_buffer.pkl', 'wb') as data_file:
        pickle.dump(data_dict, data_file)