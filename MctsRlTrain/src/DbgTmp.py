"""
@author: Fqf
@time: 20240831
@file: ParallelCollection.py
@description: Parallel deep exploration of scenes
"""

import collections
import os
import sys
import pickle
import datetime
import random
from tqdm import tqdm
import concurrent.futures
import Mcts
import DrlUtil
import GlbVar
from Dnn import PolicyValueNet
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import utils
from Util import Config
import multiprocessing

tree_info_pkl_lock = multiprocessing.Lock()
log_file_lock = multiprocessing.Lock()
collection_pbar_lock = multiprocessing.Lock()

# ==========================================================
# ======================== Function ========================
# ==========================================================

def process_row(row_idx, scene_data, policy_value_net, max_step, scene_pkl_file):
    # Initialize thread shared variables
    GlbVar.init_process_variables()
    
    scene = scene_data.iloc[row_idx]
    play_data_list = []
    # print(scene)  # [used for debug]
    # utils.plot_scene_pkl(scene_pkl_file, scene_data, row_idx, store_path=Config.StorePath.tree_info_path)   # [used for debug]

    # 2.2.1 init env
    DrlUtil.init_PcptGeo_info(scene)
    # DrlUtil.plot_PcptGeo(scene_pkl_file, row_idx) # [used for debug]

    # 2.2.2 init mcts
    MT = Mcts.MctsTree()
    DrlUtil.init_mcts_info(MT)
    SelectNodeInfo, cnt, PathFndCnt = MT.Simulate(max_step, policy_value_net)

    # 2.2.3 store tree data
    if not PathFndCnt == 0:
        state_list, V_value_list = MT.StoreTreeInfo(cnt)
        # MT.VisTree(scene_pkl_file, row_idx) # [used for debug]
        DrlUtil.plot_EnvMcts_info(scene_pkl_file, row_idx, MT)    # [used for debug]
        play_data_list = zip(state_list, V_value_list)
    else:
        with log_file_lock:
            with open(os.path.join(os.getcwd(), 'MctsRlTrain', 'output', 'log_mcts_simulation.txt'), "a") as myfile:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                myfile.write(f"{current_time} {scene.FileName} - {scene.TimeStamp} -- no path found\n") # write fail info into log
    return play_data_list

def process_chunk(chunk, scene_data, policy_value_net, max_step, scene_pkl_file, process_idx):
    chunk_results = []

    for cnt, row_idx in enumerate(chunk):
        chunk_results.extend(process_row(row_idx, scene_data, policy_value_net, max_step, scene_pkl_file))
        
    return chunk_results

def update_data_buffer(play_data_list):
    Mcts_Data_filename = f"{Config.StorePath.tree_info_path}/Mcts_Train_Data_buffer.pkl"

    with tree_info_pkl_lock:
        DataBuffer = collections.deque(maxlen=100000)
        if os.path.exists(Mcts_Data_filename):
            try:
                with open(Mcts_Data_filename, 'rb') as data_dict:
                    data_file = pickle.load(data_dict)
                    DataBuffer.extend(data_file.get('DataBuffer', []))
            except Exception as e:
                print(f"加载缓冲区时出错: {e}")
        
        DataBuffer.extend(play_data_list)
        
        data_dict = {'DataBuffer': DataBuffer}
        with open(Mcts_Data_filename, 'wb') as data_file:
            pickle.dump(data_dict, data_file)

# ==========================================================
# ======================= Collection =======================
# ==========================================================

def collection(scene_num = 100, max_step = 10000, deque_len = 300000):
    print(utils.HighLightGreenMsg('运行 CollectionMultiprocess()'))

    # ------------------------- Config -------------------------
    '''1. Load net'''
    policy_value_net = PolicyValueNet(model_file=os.path.join(Config.StorePath.tree_info_path, 'policy_value_net.pkl'))

    # --------------------- Tree Truth Gen ----------------------
    scene_file_list = DrlUtil.get_file_list_from_dir([])

    for scene_pkl_file in scene_file_list:
        '''2.1 Random scene extraction'''
        with open(scene_pkl_file, 'rb') as scene_pkl_data:   #  Select scene time slice randomly
            scene_data = pickle.load(scene_pkl_data)
            row_num = scene_data.shape[0]
            sampled_scene_idx_list = random.sample(range(row_num), min(scene_num, row_num))

    # --------------------- Tree Truth Gen ----------------------
            num_chunks = Config.MultiProcess.collection_multi_process_num
            chunk_size = len(sampled_scene_idx_list) // num_chunks
            chunks = [sampled_scene_idx_list[i:i + chunk_size] for i in range(0, len(sampled_scene_idx_list), chunk_size)]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_chunks) as executor:
                future_to_chunk = {executor.submit(process_chunk, chunk, scene_data, policy_value_net, max_step, scene_pkl_file, idx): chunk for idx, chunk in enumerate(chunks)}
                
                for future in concurrent.futures.as_completed(future_to_chunk):
                    try:
                        chunk_results = future.result()
                        update_data_buffer(chunk_results)
                    except Exception as e:
                        print(f"Error processing chunk: {e}")

    print('===== Mcts info generated done ! =====')


# -------------------------- Test ---------------------------

if __name__ == "__main__":

    collection(scene_num = 8, max_step = 100, deque_len = 100000)