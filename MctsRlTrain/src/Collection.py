"""
@author: Fqf
@time: 20240618
@file: Collection.py
@description: Used to generate training/testing datasets
"""

import collections
import os
import sys
import pickle
import random
from tqdm import tqdm
import Mcts
import DrlUtil
from Dnn import PolicyValueNet
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import utils
from Util import Config

# ==========================================================
# ======================= Collection =======================
# ==========================================================

def collection(scene_num = 100, max_step = 10000, deque_len = 300000):
    print(utils.HighLightGreenMsg('运行 collection()'))

    # ------------------------- Config -------------------------
    '''1. Load net'''
    DataBuffer = collections.deque(maxlen = deque_len)
    policy_value_net = PolicyValueNet(model_file=os.path.join(Config.StorePath.tree_info_path, 'policy_value_net.pkl'))

    # --------------------- Tree Truth Gen ----------------------
    scene_file_list = []
    scene_file_list = DrlUtil.get_file_list_from_dir(scene_file_list)

    for scene_pkl_file in scene_file_list:
        '''2.1 Random scene extraction'''
        with open(scene_pkl_file, 'rb') as scene_pkl_data:   #  Select scene time slice randomly
            scene_data = pickle.load(scene_pkl_data)
            row_num = scene_data.shape[0]
            sampled_scene_idx_list = random.sample(range(row_num), min(scene_num, row_num))

            # sampled_scene_idx_list = [3161, 3368, 4186, 1879, 2126, 3672]

            with tqdm(total=int(len(sampled_scene_idx_list)), dynamic_ncols=True, desc='Collection Progress Bar') as pbar:

                for cnt, row_idx in enumerate(sampled_scene_idx_list, start=1):
                    scene = scene_data.iloc[row_idx]
                    # print(scene)  # [used for debug]
                    # utils.plot_scene_pkl(scene_pkl_file, scene_data, row_idx, store_path=Config.StorePath.tree_info_path)   # [used for debug]

                    '''2.2 Generating truth tree'''
                    # 2.2.1 init env
                    DrlUtil.init_PcptGeo_info(scene)    # (collision free with SP and TP)
                    # DrlUtil.plot_PcptGeo(scene_pkl_file, row_idx) # [used for debug]

                    # 2.2.2 init mcts
                    MT = Mcts.MctsTree()
                    DrlUtil.init_mcts_info(MT)
                    sim_info = MT.Simulate(max_step, policy_value_net)

                    # 2.2.3 store tree data
                    state_list, V_value_list = MT.StoreTreeInfo(sim_info)
                    # MT.VisTree(scene_pkl_file, row_idx) # [used for debug]
                    DrlUtil.plot_EnvMcts_info(scene_pkl_file, row_idx, MT)    # [used for debug]
                    play_data_list = zip(state_list, V_value_list)  # Each element of the play_data_list list is a tuple containing a state, act_probs and value

                    '''2.3 Store mcts tree information'''
                    Mcts_Data_filename = f"{Config.StorePath.tree_info_path}/Mcts_Train_Data_buffer.pkl"
                    if os.path.exists(Mcts_Data_filename):
                        try:
                            with open(Mcts_Data_filename, 'rb') as data_dict:
                                data_file = pickle.load(data_dict)  # This is a dictionary
                                DataBuffer = collections.deque(maxlen = 100000) # Keep the most recently added elements, and remove the oldest elements
                                DataBuffer.extend(data_file['DataBuffer'])
                                del data_file
                                DataBuffer.extend(play_data_list)
                            # print('-- Import data from buffer_pkl success !')
                        except:
                            print(utils.HighLightRedMsg('Import data from buffer_pkl fail !'))
                    else:
                        DataBuffer.extend(play_data_list)
                    
                    data_dict = {'DataBuffer': DataBuffer}
                    with open(Mcts_Data_filename, 'wb') as data_file:
                        pickle.dump(data_dict, data_file)

                    '''3. Progress Bar'''
                    cycle_interval = 1
                    if cnt % cycle_interval == 0:
                            pbar.set_postfix({
                                'episode': '%d' % (cnt)
                                })
                            
                            pbar.update(cycle_interval)

    print('===== Mcts info generated done ! =====')

# -------------------------- Test ---------------------------

if __name__ == "__main__":

    collection(scene_num = 100, max_step = 10000, deque_len = 100000)