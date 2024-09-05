"""
@author: Fqf
@time: 20240817
@file: Treeinfo2Dataset.py
@description: Used to generate Img & Label datasets
"""

import pickle
import time
import random
import shutil
import DrlUtil
import os
import sys
from tqdm import tqdm
import concurrent.futures
import multiprocessing
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import Config
from Util import utils

Convert2DataSet_pbar_lock = multiprocessing.Lock()

# ==========================================================
# ======================= GenDataSet =======================
# ==========================================================

def process_chunk(chunk, w):
    for cnt, scene in enumerate(chunk):
        DrlUtil.GenImgLabel(scene, Config.StorePath.train_dataset_path)

        if cnt == len(chunk) - 1:
            DrlUtil.flush_cache()

    w.send(1)

def Convert2DataSet(sample_size=100000):
    print(utils.HighLightGreenMsg('运行 Convert2DataSet()'))

    # 1. Clean old dataset
    csv_path = os.path.join(Config.StorePath.train_dataset_path, 'label.csv')
    img_path = os.path.join(Config.StorePath.train_dataset_path, 'images')

    if os.path.isfile(csv_path):
        os.remove(csv_path)
    if os.path.isdir(img_path):
        shutil.rmtree(img_path)

    time.sleep(5)
    os.makedirs(img_path, exist_ok=True)

    # 2. Gen dataset
    TreeInfoPkl = os.path.join(Config.StorePath.tree_info_path, 'Mcts_Train_Data_buffer.pkl')

    with open(TreeInfoPkl, 'rb') as data_dict:
        data_file = pickle.load(data_dict)  # This is a dictionary
        scene_data = data_file['DataBuffer']

        sampled_indices = random.sample(scene_data, min(len(scene_data), sample_size))
        num_chunks = Config.MultiProcess.Convert2DataSet_multi_precess_num
        chunks = [sampled_indices[i::num_chunks] for i in range(num_chunks)]
        
        with tqdm(total=int(len(sampled_indices)), dynamic_ncols=True, desc='Convert2DataSet Progress Bar') as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_chunks) as executor:
                
                r,w = multiprocessing.Pipe(duplex=False)
                
                futures = [executor.submit(process_chunk, chunk, w) for chunk in enumerate(chunks)]
                
                cnt=0
                while cnt<int(len(sampled_indices)):
                    try:
                        msg=r.recv()
                        cnt+=1

                        cycle_interval = 1
                        if cnt % cycle_interval == 0 or cnt == len(sampled_indices) - 1:
                            # pbar.set_postfix({'thread_idx': f'{thread_idx}'})
                            pbar.update(cycle_interval)

                    except EOFError:
                        break

                # Wait for all futures to complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()  # Raise exception if the task failed
                    except Exception as e:
                        print(utils.HighLightRedMsg(f"Convert2DataSet err: {e}"))

    print('===== Convert2DataSet done ! =====')


# -------------------------- Test ---------------------------
if __name__ == '__main__':
    Convert2DataSet(sample_size=100000)