"""
@author: Fqf
@time: 20240831
@file: ParallelCollection.py
@description: Parallel deep exploration of scenes
"""

import collections
import os
import sys
import gc
import torch
import pickle
import datetime
import random
from tqdm import tqdm
import time
import Mcts
import DrlUtil
from Dnn import PolicyValueNet
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import utils
from Util import Config
import multiprocessing

tree_info_pkl_lock = multiprocessing.Lock()
log_file_lock = multiprocessing.Lock()
cycle_interval = 1

# ==========================================================
# ======================== Function ========================
# ==========================================================

def process_row(row_idx, scene_data, policy_value_net, max_step, scene_pkl_file):

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
    _, cnt, PathFndCnt = MT.Simulate(max_step, policy_value_net)

    # 2.2.3 store tree data
    if not (PathFndCnt == 0):
        state_list, V_value_list = MT.StoreTreeInfo(cnt)
        # MT.VisTree(scene_pkl_file, row_idx) # [used for debug]
        DrlUtil.plot_EnvMcts_info(scene_pkl_file, row_idx, True)    # [used for debug]
        play_data_list = zip(state_list, V_value_list)
    else:
        DrlUtil.plot_EnvMcts_info(scene_pkl_file, row_idx, False)    # [used for debug]
        with log_file_lock:
            with open(os.path.join(os.getcwd(), 'MctsRlTrain', 'output', 'log_mcts_simulation.txt'), "a") as myfile:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                myfile.write(f"{current_time} {scene.FileName} - {scene.TimeStamp} -- no path found\n") # write fail info into log
    return play_data_list

def batch_exec(chunk, scene_data, policy_value_net, max_step, scene_pkl_file, w):
    chunk_results = collections.deque(maxlen=100000)
    batch_size = 2

    for idx, row_idx in enumerate(chunk):
        chunk_results.extend(process_row(row_idx, scene_data, policy_value_net, max_step, scene_pkl_file))

        if (idx + 1) % batch_size == 0 or (idx + 1) == len(chunk):
            update_data_buffer(chunk_results)
            chunk_results.clear()

        if (idx + 1) % cycle_interval == 0:
            w.send(cycle_interval)
    if len(chunk) % cycle_interval != 0:
        w.send(len(chunk) % cycle_interval)

    return True

def update_data_buffer(chunk_results):
    Mcts_Data_filename = f"{Config.StorePath.tree_info_path}/Mcts_Train_Data_buffer.pkl"

    with tree_info_pkl_lock:
        DataBuffer = collections.deque(maxlen=100000)

        if os.path.exists(Mcts_Data_filename):
            write_cnt = 0
            while write_cnt < 3:
                try:
                    write_cnt = write_cnt + 1
                    with open(Mcts_Data_filename, 'rb') as data_dict:
                        data_file = pickle.load(data_dict)
                        DataBuffer.extend(data_file.get('DataBuffer', []))
                        break
                except Exception as e:
                    if write_cnt == 3:
                        print(utils.HighLightRedMsg(f"尝试3次加载缓冲区均出错: {e}"))
                        break
                    time.sleep(30)
        
        DataBuffer.extend(chunk_results)
        data_dict = {'DataBuffer': DataBuffer}
        with open(Mcts_Data_filename, 'wb') as data_file:
            pickle.dump(data_dict, data_file)

def on_success(result):
    pass
    # print(f"Result: {result}")

def on_error(exception):
    print(f"Multi-Task failed with exception: {exception}")

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
            # sampled_scene_idx_list = [3161, 3368, 4186, 1879, 2126, 3672]

    # ---------------------- Multi execute ----------------------
            num_chunks = Config.MultiProcess.collection_multi_process_num
            chunk_size = len(sampled_scene_idx_list) // num_chunks
            chunks = [sampled_scene_idx_list[i:i + chunk_size] for i in range(0, len(sampled_scene_idx_list), chunk_size)]
            pool_size = num_chunks

            ''' multiprocessing.Pool '''
            with tqdm(total=int(len(sampled_scene_idx_list)), dynamic_ncols=True, desc='Collection Progress Bar') as pbar:
                with multiprocessing.Pool(processes=pool_size) as pool:
                    
                    async_results = []
                    r,w = multiprocessing.Pipe(duplex=False)

                    for chunk in chunks:
                        async_result  = pool.apply_async(batch_exec, 
                                                        args=(chunk, scene_data, policy_value_net, max_step, scene_pkl_file, w),
                                                        callback=on_success, error_callback=on_error)
                        async_results.append(async_result)

                    cnt=0
                    while cnt<int(len(sampled_scene_idx_list)):
                        try:
                            msg=r.recv()
                            cnt+=msg
                            pbar.update(msg)
                        except EOFError:
                            break

                    for result in async_results:
                        result.wait()

    # Release computing resources
    gc.collect()
    torch.cuda.empty_cache()

    print('===== Mcts info generated done ! =====')


# -------------------------- Test ---------------------------

if __name__ == "__main__":

    Config.MultiProcess.collection_multi_process_num = 6
    collection(scene_num = 20, max_step = 8000, deque_len = 100000)