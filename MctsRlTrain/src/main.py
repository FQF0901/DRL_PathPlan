"""
@author: Fqf
@time: 20240807
@file: main.py
@description: main func of RL
"""

import os
import sys
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
    
def run_functions():
    with concurrent.futures.ThreadPoolExecutor() as executor:

        # 1. Train net
        net_model = os.path.join(Config.StorePath.train_dataset_path, 'policy_value_net.pkl')
        csv_path = os.path.join(Config.StorePath.train_dataset_path, 'label.csv')
        img_path = os.path.join(Config.StorePath.train_dataset_path, 'images')

        training_pipeline = TrainPipeline(init_model=net_model)       
        future_train = executor.submit(training_pipeline.run, csv_file=csv_path, img_folder=img_path)
        
        # 2. Collection
        future_collection = executor.submit(collection, scene_num=100, max_step=10000, deque_len=300000)

        concurrent.futures.wait([future_collection, future_train])


# ==========================================================
# ======================== main fun ========================
# ==========================================================

if __name__ == "__main__":

    # 1. Init net
    policy_value_net_path = os.path.join(os.path.join(os.getcwd(), 'MctsRlTrain', 'output'), 'policy_value_net.pkl')

    if not os.path.isfile(policy_value_net_path):
        policy_value_net = PolicyValueNet()
        policy_value_net.save_model(model_file = policy_value_net_path)
        print(utils.HighLightRedMsg('模型路径不存在，从零开始训练'))

    else:
        print(utils.HighLightGreenMsg('已加载上次最终模型'))

    shutil.copy(policy_value_net_path, Config.StorePath.train_dataset_path)
    shutil.copy(policy_value_net_path, Config.StorePath.tree_info_path)

    # 2. Generate new scenes for initial training
    collection(scene_num = 10, max_step = 10000, deque_len = 300000)
    Treeinfo2Dataset.Convert2DataSet()
    update_input_files()

    time.sleep(3)

    # 3. Start the formal loop (based on the initialized or old net parameter)
    for _ in range(1):
        run_functions()

        Treeinfo2Dataset.Convert2DataSet()
        update_input_files()

        time.sleep(10)

    # try:
    #     os.system('rundll32.exe powrprof.dll,SetSuspendState 0,1,0')
    #     # os.system("shutdown /s /t 0")
    # except Exception as e:
    #     print(f"An error occurred: {e}")