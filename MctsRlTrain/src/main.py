"""
@author: Fqf
@time: 20240807
@file: main.py
@description: main func of RL
"""

import os
import re
import sys
import shutil
import time
import Treeinfo2Dataset
# from Collection import collection
from MctsRlTrain.src.CollectionMultiprocess import collection
from Dnn import PolicyValueNet
from Train import TrainPipeline
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import utils
from Util import Config

# ==========================================================
# ======================== Function ========================
# ==========================================================

def clear_path(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
            except Exception as e:
                print(utils.HighLightRedMsg(f"Error while deleting {file_path}: {e}"))

def store_2_pkl_files(only_tree_info_pkl=False):
    # 1. train_dataset_path
    if not only_tree_info_pkl:
        output_path = os.path.join(os.getcwd(), 'MctsRlTrain', 'output')
        
        files = [f for f in os.listdir(Config.StorePath.train_dataset_path) if f.startswith('policy_value_net_') and f.endswith('.pkl')]
        
        max_x = -1
        max_file = None
        pattern = re.compile(r'policy_value_net_(\d+)\.pkl')
        
        for file in files:
            match = pattern.match(file)
            if match:
                x = int(match.group(1))
                if x > max_x:
                    max_x = x
                    max_file = file
        
        if max_file:
            src_path = os.path.join(Config.StorePath.train_dataset_path, max_file)
            dst_path = os.path.join(output_path, 'policy_value_net.pkl')
            shutil.copy(src_path, dst_path)
            print(f"文件 {src_path} 已复制并重命名为 {dst_path}")
        else:
            print(f"没有找到符合条件的文件")

    # 2. tree_info_path
    scene_in_tree_folder = os.path.join(Config.StorePath.tree_info_path, 'Mcts_Train_Data_buffer.pkl')
    
    if os.path.isfile(scene_in_tree_folder):
        shutil.copy(scene_in_tree_folder, os.path.join(os.getcwd(), 'MctsRlTrain', 'output'))
    else:
        print(f"文件 {scene_in_tree_folder} 不存在")

def main(main_for_loop_num, start_from_train_or_collection,
             collection_scene_num, collection_max_step, 
             train_batch_size, train_epoch_num):
    
    # 1. Clean folder
    clear_path(Config.StorePath.train_dataset_path)
    clear_path(Config.StorePath.tree_info_path)
    clear_path(os.path.join(os.getcwd(), 'logs'))

    # 2. Init net.pkl
    policy_value_net_pkl = os.path.join(os.path.join(os.getcwd(), 'MctsRlTrain', 'output'), 'policy_value_net.pkl')

    if not os.path.isfile(policy_value_net_pkl):
        policy_value_net = PolicyValueNet()
        policy_value_net.save_model(model_file = policy_value_net_pkl)
        print(utils.HighLightRedMsg(f'{policy_value_net_pkl}不存在, 从零开始训练'))

    else:
        print(utils.HighLightGreenMsg('加载上次最终policy_value_net.pkl'))

    # 2. Init tree_info.pkl
    if start_from_train_or_collection == 1:
        # 2.1 Prepare pkl file
        tree_info_pkl = os.path.join(os.path.join(os.getcwd(), 'MctsRlTrain', 'output'), 'Mcts_Train_Data_buffer.pkl')
        shutil.copy(tree_info_pkl, Config.StorePath.tree_info_path)

        # 2.2 Gen dataset
        Treeinfo2Dataset.Convert2DataSet(sample_size=100000)
        
        # 2.3 Train
        shutil.copy(policy_value_net_pkl, Config.StorePath.train_dataset_path)
        training_pipeline = TrainPipeline(init_model=os.path.join(Config.StorePath.train_dataset_path, 'policy_value_net.pkl'), 
                                        batch_size=train_batch_size,
                                        epoch_num=train_epoch_num)       
        training_pipeline.run(csv_file=os.path.join(Config.StorePath.train_dataset_path, 'label.csv'), 
                            img_folder=os.path.join(Config.StorePath.train_dataset_path, 'images'))
        
        # 2.4 Store files and reset folders
        store_2_pkl_files(only_tree_info_pkl=True)
        clear_path(Config.StorePath.train_dataset_path)
        clear_path(Config.StorePath.tree_info_path)

    # 3. Start the formal loop (based on the initialized or old net parameter)
    for _ in range(main_for_loop_num):
        # 3.0 Prepare pkl file
        shutil.copy(policy_value_net_pkl, Config.StorePath.train_dataset_path)
        shutil.copy(policy_value_net_pkl, Config.StorePath.tree_info_path)

        # 3.1 Gen raw dataset
        collection(scene_num=collection_scene_num, 
                   max_step=collection_max_step, deque_len=300000)

        # 3.2 Gen dataset
        Treeinfo2Dataset.Convert2DataSet(sample_size=100000)
        
        # 3.3 Train
        training_pipeline = TrainPipeline(init_model=os.path.join(Config.StorePath.train_dataset_path, 'policy_value_net.pkl'), 
                                          batch_size=train_batch_size,
                                          epoch_num=train_epoch_num)       
        training_pipeline.run(csv_file=os.path.join(Config.StorePath.train_dataset_path, 'label.csv'), 
                            img_folder=os.path.join(Config.StorePath.train_dataset_path, 'images'))
        
        # 3.4 Store files and reset folders
        store_2_pkl_files(only_tree_info_pkl=False)
        clear_path(Config.StorePath.train_dataset_path)
        clear_path(Config.StorePath.tree_info_path)

# ==========================================================
# ======================== main fun ========================
# ==========================================================

if __name__ == "__main__":
    try:
        # 1. Config
        Config.MultiProcess.collection_multi_process_num = 6
        Config.MultiProcess.Convert2DataSet_multi_precess_num = 6
        
        main_for_loop_num = 5
        start_from_train_or_collection = 0  # 1: start from train, others: start from collection

        collection_scene_num = 200
        collection_max_step = 8000

        train_batch_size = 32
        train_epoch_num = 10

        # 2. main func
        main(main_for_loop_num, start_from_train_or_collection,
             collection_scene_num, collection_max_step, 
             train_batch_size, train_epoch_num)

        print(utils.HighLightGreenMsg('The entire process is completed !'))

    except KeyboardInterrupt:
        print(utils.HighLightRedMsg('\n\rQuit'))


    # 4. Sleep computer
    # try:
    #     time.sleep(30)
    #     os.system('rundll32.exe powrprof.dll,SetSuspendState 0,1,0')
    #     # os.system("shutdown /s /t 0")
    # except Exception as e:
    #     print(utils.HighLightRedMsg(f"An error occurred: {e}"))