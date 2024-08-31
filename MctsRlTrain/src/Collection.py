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
import time
from tqdm import tqdm
import Mcts
import MctsEfct
import DrlUtil
import DrlCfg
from Dnn import PolicyValueNet
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import utils
from Util import Config

# ==========================================================
# ======================= Collection =======================
# ==========================================================

# ------------------------- Config -------------------------
max_step_each_epsd = 10000
DataBuffer = collections.deque(maxlen = 100000)

# 1. Load net
policy_value_net = PolicyValueNet(model_file=os.path.join(Config.StorePath.net_path, 'policy_value_net.pkl'))

# --------------------- Tree Truth Gen ----------------------
scene_file_list = []
scene_file_list = DrlUtil.get_file_list_from_dir(scene_file_list)

for scene_pkl_file in scene_file_list:
    # 2.1 Random scene extraction
    with open(scene_pkl_file, 'rb') as scene_pkl_data:   #  Select scene time slice randomly
        scene_data = pickle.load(scene_pkl_data)
        row_num = scene_data.shape[0]
        sampled_scene_idx_list = random.sample(range(row_num), min(200, row_num))

        # sampled_scene_idx_list = [3161, 3368, 4186, 1879, 2126, 3677]

        with tqdm(total=int(len(sampled_scene_idx_list)), dynamic_ncols=True, desc='Progress Bar') as pbar:

            for cnt, row_idx in enumerate(sampled_scene_idx_list, start=1):
                scene = scene_data.iloc[row_idx]
                # print(scene)  # [used for debug]
                # utils.plot_scene_pkl(scene_pkl_file, scene_data, row_idx, store_path=Config.StorePath.tree_info_path)   # [used for debug]

                # 2.2 Generating truth tree
                if DrlCfg.TreePara.TreeType == 1:   # 1: Mcts, 2: Mcts efficient, 3: Mcts Bi-direction
                    # 2.2.1 init env
                    # start_time = time.time()
                    DrlUtil.init_PcptGeo_info(scene)    # (collision free with SP and TP)
                    # DrlUtil.plot_PcptGeo(scene_pkl_file, row_idx) # [used for debug]

                    # 2.2.2 init mcts
                    MT = Mcts.MctsTree()
                    DrlUtil.init_mcts_info(MT)
                    sim_info = MT.Simulate(max_step_each_epsd, policy_value_net)

                    # 2.2.3 store tree data
                    state_list, V_value_list = MT.StoreTreeInfo(sim_info)
                    MT.VisTree(scene_pkl_file, row_idx) # [used for debug]
                    DrlUtil.plot_EnvMcts_info(scene_pkl_file, row_idx, MT)    # [used for debug]
                    play_data_list = zip(state_list, V_value_list)  # Each element of the play_data_list list is a tuple containing a state, act_probs and value

                elif DrlCfg.TreePara.TreeType == 2:
                    MT = MctsEfct.MctsEfctTree()
                    DrlUtil.init_mctsefct_info(MT)

                    DrlUtil.init_PcptGeo_info(scene)
                    # DrlUtil.plot_PcptGeo(scene_pkl_file, row_idx) # [used for debug]

                    MT.Simulate(max_step_each_epsd)
                    MT.VisTree(scene_pkl_file, row_idx)

                    state_list, V_value_list = MT.StoreTreeInfo()

                    play_data_list = zip(state_list, V_value_list)

                # 2.3 Store mcts tree information
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

                # end_time = time.time()

                # 3. Progress Bar
                cycle_interval = 1
                if cnt % cycle_interval == 0:
                        pbar.set_postfix({
                            'episode': '%d' % (cnt)
                            })
                        
                        pbar.update(cycle_interval)

                # print(f"Single scene time consumption: {end_time - start_time:.4f} s")
                # max_step_each_epsd = 50
                # init_PcptGeo_info: 0.0025 s, init_mcts_info: 0.0000 s, mcts_simulate: 68.8811 s, plot_EnvMcts_info: 0.0945 s, store_time: 0.0015 s

print('===== Mcts info generated done ! =====')

# time.sleep(60)
# try:
#     os.system('rundll32.exe powrprof.dll,SetSuspendState 0,1,0')
#     # os.system("shutdown /s /t 0")
# except Exception as e:
#     print(f"An error occurred: {e}")

# assert path.L >= 0.01