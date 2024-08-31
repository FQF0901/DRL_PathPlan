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
import threading
from tqdm import tqdm
import concurrent.futures
import Mcts
import MctsEfct
import DrlUtil
import DrlCfg
from Dnn import PolicyValueNet
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import utils
from Util import Config

# ==========================================================
# ======================== Function ========================
# ==========================================================

def process_row(row_idx):
    scene = scene_data.iloc[row_idx]
    
    play_data_list = []
    if DrlCfg.TreePara.TreeType == 1:
        DrlUtil.init_PcptGeo_info(scene)
        MT = Mcts.MctsTree()
        DrlUtil.init_mcts_info(MT)
        MT.Simulate(max_step_each_epsd, policy_value_net)
        state_list, V_value_list = MT.StoreTreeInfo()
        MT.VisTree(scene_pkl_file, row_idx)
        DrlUtil.plot_EnvMcts_info(scene_pkl_file, row_idx, MT)
        play_data_list = zip(state_list, V_value_list)
    
    elif DrlCfg.TreePara.TreeType == 2:
        MT = MctsEfct.MctsEfctTree()
        DrlUtil.init_mctsefct_info(MT)
        DrlUtil.init_PcptGeo_info(scene)
        MT.Simulate(max_step_each_epsd)
        MT.VisTree(scene_pkl_file, row_idx)
        state_list, V_value_list = MT.StoreTreeInfo()
        play_data_list = zip(state_list, V_value_list)

    return play_data_list

def process_chunk(chunk):
    chunk_results = []
    for row_idx in chunk:
        chunk_results.extend(process_row(row_idx))
    return chunk_results

def update_data_buffer(play_data_list):
    Mcts_Data_filename = f"{Config.StorePath.tree_info_path}/Mcts_Train_Data_buffer.pkl"
    
    if os.path.exists(Mcts_Data_filename):
        try:
            with open(Mcts_Data_filename, 'rb') as data_dict:
                data_file = pickle.load(data_dict)
                DataBuffer = collections.deque(maxlen=100000)
                DataBuffer.extend(data_file['DataBuffer'])
                del data_file
                DataBuffer.extend(play_data_list)
        except:
            print('Import data from buffer_pkl fail !')
    else:
        DataBuffer = collections.deque(play_data_list, maxlen=100000)
    
    data_dict = {'DataBuffer': DataBuffer}
    with open(Mcts_Data_filename, 'wb') as data_file:
        pickle.dump(data_dict, data_file)

# ==========================================================
# ======================= Collection =======================
# ==========================================================

if __name__ == "__main__":

# ------------------------- Config -------------------------
    max_step_each_epsd = 12
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
            sampled_scene_idx_list = [174, 734, 1024, 1167, 2872, 3375, 4234, 5084]
# --------------------- Tree Truth Gen ----------------------
            lock = threading.Lock()

            # Determine the chunk size for splitting the workload
            num_chunks = 4
            chunk_size = len(sampled_scene_idx_list) // num_chunks
            
            # Split the workload into chunks
            chunks = [sampled_scene_idx_list[i:i + chunk_size] for i in range(0, len(sampled_scene_idx_list), chunk_size)]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as executor:
                # process_chunk 是处理每个数据块的函数，而 chunk 是要处理的数据块
                # 返回一个Future 对象表示异步执行的结果
                future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
                
                # 遍历这些完成的任务，并依次处理它们的结果
                for future in concurrent.futures.as_completed(future_to_chunk):
                    try:
                        chunk_results = future.result()
                        # Lock to ensure thread-safe access to file
                        with lock:
                            update_data_buffer(chunk_results)
                    except Exception as e:
                        print(f"Error processing chunk: {e}")

    print('===== Mcts info generated done ! =====')