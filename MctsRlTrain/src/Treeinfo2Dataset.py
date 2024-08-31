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
    csv_path = os.path.join(Config.StorePath.train_dataset_path, 'label.csv')
    img_path = os.path.join(Config.StorePath.train_dataset_path, 'images')

    if os.path.isfile(csv_path):
        os.remove(csv_path)

    if os.path.isdir(img_path):
        shutil.rmtree(img_path)

    os.makedirs(img_path, exist_ok=True)

    time.sleep(3)

    # 2. Gen dataset
    TreeInfoPkl = os.path.join(Config.StorePath.tree_info_path, 'Mcts_Train_Data_buffer.pkl')

    with open(TreeInfoPkl, 'rb') as data_dict:
        data_file = pickle.load(data_dict)  # This is a dictionary

        for scene in data_file['DataBuffer']:
            DrlUtil.GenImgLabel(scene, Config.StorePath.train_dataset_path)


# -------------------------- Test ---------------------------
if __name__ == '__main__':
    Convert2DataSet()