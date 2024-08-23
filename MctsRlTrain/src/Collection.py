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
import Mcts
import MctsEfct
import DrlUtil
import DrlCfg
from Dnn import PolicyValueNet
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import utils

# ==========================================================
# ======================= Collection =======================
# ==========================================================

# ------------------------- Config -------------------------
max_step_each_epsd = 10000
DataBuffer = collections.deque(maxlen = 100000)

# 1.1 scene pkl file location
scene_pkl_path = r'D:\DataSet\TimeSliceData'    # [ Config ]: The folder address where scene data is located

# 1.2 tree pkl file location
tree_pkl_path = r'D:\DataSet\TreeinfoData'

# 1.3 Load net
mcts_value_net = PolicyValueNet(model_file=r'C:\01_Project\10_Git\PECU_DRL\MctsRlTrain\TrainDataSet\crnt_policy_value_net.pkl')   # used in DrlUtil.py

# --------------------- Tree Truth Gen ----------------------
scene_file_list = []
scene_file_list = DrlUtil.get_file_list_from_dir(scene_pkl_path, scene_file_list)

for scene_pkl_file in scene_file_list:
    # 2.1 Random scene extraction
    with open(scene_pkl_file, 'rb') as scene_pkl_data:   #  Select scene time slice randomly
        scene_data = pickle.load(scene_pkl_data)
        row_num = scene_data.shape[0]
        
        for row_idx in range(0, row_num):
            scene = scene_data.iloc[row_idx]
            # print(scene)  # [used for debug]
            # utils.plot_scene_pkl(scene_pkl_file, scene_data, row_idx, store_path=tree_pkl_path)   # [used for debug]

            # 2.2 Generating truth tree
            if DrlCfg.TreePara.TreeType == 1:   # 1: Mcts, 2: Mcts efficient, 3: Mcts Bi-direction
                # 2.2.1 init env
                DrlUtil.init_PcptGeo_info(scene)
                # DrlUtil.plot_PcptGeo(scene_pkl_file, row_idx, tree_pkl_path) # [used for debug]

                # 2.2.2 init mcts
                MT = Mcts.MctsTree()
                DrlUtil.init_mcts_info(MT, scene)
                MT.Simulate(max_step_each_epsd)
                MT.VisTree(scene_pkl_file, row_idx, store_path=tree_pkl_path)

                # 2.2.3 store tree data
                state_list, V_value_list = MT.StoreTreeInfo()
                play_data_list = zip(state_list, V_value_list)  # Each element of the play_data_list list is a tuple containing a state, act_probs and value

            elif DrlCfg.TreePara.TreeType == 2:
                MT = MctsEfct.MctsEfctTree()
                DrlUtil.init_mctsefct_info(MT, scene)

                DrlUtil.init_PcptGeo_info(scene)
                # DrlUtil.plot_PcptGeo(scene_pkl_file, row_idx, tree_pkl_path) # [used for debug]

                MT.Simulate(max_step_each_epsd)
                MT.VisTree(scene_pkl_file, row_idx, store_path=tree_pkl_path)

                state_list, V_value_list = MT.StoreTreeInfo()

                play_data_list = zip(state_list, V_value_list)

            # 2.3 Store mcts tree information
            Mcts_Data_filename = f"{tree_pkl_path}/Mcts_Train_Data_buffer.pkl"
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