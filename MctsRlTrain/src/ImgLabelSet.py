"""
@author: Fqf
@time: 20240817
@file: ImgLabelSet.py
@description: Used to generate Img & Label datasets
"""

import pickle
import DrlUtil
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import utils
from Util import Config

TreeInfoPkl = os.path.join(Config.StorePath.tree_info_path, 'Mcts_Train_Data_buffer.pkl')

with open(TreeInfoPkl, 'rb') as data_dict:
    data_file = pickle.load(data_dict)  # This is a dictionary

    for scene in data_file['DataBuffer']:
        DrlUtil.GenImgLabel(scene, Config.StorePath.train_dataset_path)