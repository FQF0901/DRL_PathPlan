"""
@author: Fqf
@time: 20240807
@file: main.py
@description: main func of RL
"""

import os
import sys
import glob
import shutil
import concurrent.futures
import time
import Treeinfo2Dataset
from Collection import collection
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
                print(f"Error while deleting {file_path}: {e}")

def delete_files_for_loop(tree_info_path, train_dataset_path):
    # 1. Delete. png and. svg files, as well as Mcts_Train_data-buffer. pkl
    for extension in ['*.png', '*.svg']:
        for filepath in glob.glob(os.path.join(tree_info_path, extension)):
            os.remove(filepath)
    
    pkl_file = os.path.join(tree_info_path, 'Mcts_Train_Data_buffer.pkl')
    if os.path.exists(pkl_file):
        os.remove(pkl_file)
    
    # 2. Delete files containing 'policy-value_net_' with a suffix of. pkl
    for filepath in glob.glob(os.path.join(train_dataset_path, 'policy_value_net_*.pkl')):
        os.remove(filepath)

def update_input_files():
    # 1. train_dataset_path
    net_in_train_folder = os.path.join(Config.StorePath.train_dataset_path, 'policy_value_net.pkl')
    if os.path.isfile(net_in_train_folder):
        shutil.copy(net_in_train_folder, os.path.join(os.path.join(os.getcwd(), 'MctsRlTrain', 'output'), 'policy_value_net.pkl'))
        shutil.copy(net_in_train_folder, Config.StorePath.tree_info_path)
    else:
        print(f"文件 {net_in_train_folder} 不存在")

    # 2. tree_info_path
    scene_in_tree_folder = os.path.join(Config.StorePath.tree_info_path, 'Mcts_Train_Data_buffer.pkl')
    if os.path.isfile(scene_in_tree_folder):
        shutil.copy(scene_in_tree_folder, os.path.join(os.getcwd(), 'MctsRlTrain', 'output'))
        shutil.copy(scene_in_tree_folder, Config.StorePath.train_dataset_path)
    else:
        print(f"文件 {scene_in_tree_folder} 不存在")
    
def multi_threaded_func():
    with concurrent.futures.ThreadPoolExecutor() as executor:

        # 1. Train net
        net_model = os.path.join(Config.StorePath.train_dataset_path, 'policy_value_net.pkl')
        csv_path = os.path.join(Config.StorePath.train_dataset_path, 'label.csv')
        img_path = os.path.join(Config.StorePath.train_dataset_path, 'images')

        training_pipeline = TrainPipeline(init_model=net_model)       
        training_pipeline.epoch_num = 1000
        training_pipeline.epochs = 1
        future_train = executor.submit(training_pipeline.run, csv_file=csv_path, img_folder=img_path)
        
        # 2. Collection
        future_collection = executor.submit(collection, scene_num=100, max_step=15000, deque_len=300000)

        concurrent.futures.wait([future_collection, future_train])


# ==========================================================
# ======================== main fun ========================
# ==========================================================

if __name__ == "__main__":

    # 1. Clean folder
    clear_path(Config.StorePath.train_dataset_path)
    clear_path(Config.StorePath.tree_info_path)

    # 2. Init env
    policy_value_net_pkl = os.path.join(os.path.join(os.getcwd(), 'MctsRlTrain', 'output'), 'policy_value_net.pkl')

    if not os.path.isfile(policy_value_net_pkl):
        policy_value_net = PolicyValueNet()
        policy_value_net.save_model(model_file = policy_value_net_pkl)
        print(utils.HighLightRedMsg(f'{policy_value_net_pkl}不存在, 从零开始训练'))

    else:
        print(utils.HighLightGreenMsg('加载上次最终{policy_value_net_pkl}'))

    shutil.copy(policy_value_net_pkl, Config.StorePath.train_dataset_path)
    shutil.copy(policy_value_net_pkl, Config.StorePath.tree_info_path)

    # 3. Generate new scenes for initial training
    collection(scene_num = 50, max_step = 15000, deque_len = 300000)
    Treeinfo2Dataset.Convert2DataSet()
    update_input_files()

    # 4. Start the formal loop (based on the initialized or old net parameter)
    for _ in range(5):
        delete_files_for_loop(Config.StorePath.tree_info_path, Config.StorePath.train_dataset_path)    # Clear PNG, SVG and Mcts_Train_Data_buffer.pkl, policy_value_net_n.pkl

        multi_threaded_func()   # Multi threaded parallel computing main function

        Treeinfo2Dataset.Convert2DataSet()
        update_input_files()

    # 5. Sleep computer
    try:
        time.sleep(30)
        os.system('rundll32.exe powrprof.dll,SetSuspendState 0,1,0')
        # os.system("shutdown /s /t 0")
    except Exception as e:
        print(f"An error occurred: {e}")