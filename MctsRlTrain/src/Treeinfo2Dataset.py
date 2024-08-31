"""
@author: Fqf
@time: 20240817
@file: Treeinfo2Dataset.py
@description: Used to generate Img & Label datasets
"""

import pickle
import time
import shutil
import DrlUtil
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import Config

def Convert2DataSet():
    # 1. Clean old dataset
    for filename in os.listdir(Config.StorePath.train_dataset_path):
        file_path = os.path.join(Config.StorePath.train_dataset_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    images_folder_path = os.path.join(Config.StorePath.train_dataset_path, 'images')
    os.makedirs(images_folder_path, exist_ok=True)

    time.sleep(5)

    # 2. Gen dataset
    TreeInfoPkl = os.path.join(Config.StorePath.tree_info_path, 'Mcts_Train_Data_buffer.pkl')

    with open(TreeInfoPkl, 'rb') as data_dict:
        data_file = pickle.load(data_dict)  # This is a dictionary

        for scene in data_file['DataBuffer']:
            DrlUtil.GenImgLabel(scene, Config.StorePath.train_dataset_path)


# -------------------------- Test ---------------------------
if __name__ == '__main__':
    Convert2DataSet()