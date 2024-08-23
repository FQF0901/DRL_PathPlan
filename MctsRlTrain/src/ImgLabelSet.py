"""
@author: Fqf
@time: 20240817
@file: ImgLabelSet.py
@description: Used to generate Img & Label datasets
"""

import pickle
import DrlUtil

TreeInfoPkl = r'D:\DataSet\TreeinfoData\Mcts_Train_Data_buffer.pkl'
Img_Label_path = r'D:\DataSet\TrainDataSet'

with open(TreeInfoPkl, 'rb') as data_dict:
    data_file = pickle.load(data_dict)  # This is a dictionary

    for scene in data_file['DataBuffer']:
        DrlUtil.GenImgLabel(scene, Img_Label_path)