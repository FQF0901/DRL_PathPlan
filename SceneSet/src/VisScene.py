"""
@author: fqf
@time: 20240722
@file: VisData.py
@description: Visualization for mf4 data
"""

import matplotlib.pyplot as plt
import pickle
import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import utils

# ==========================================================
# ===================== visualization ======================
# ==========================================================

scene_pkl_file_path = r'D:\DataSet\TimeSliceData'
scene_pkl_file_name = r'\mf4_time_slice_data_part1.pkl'
VisMod = 3  # 1:Random Drawing, 2:Specify idx to draw, 3:All drawings

with open(scene_pkl_file_path+scene_pkl_file_name, 'rb') as scene_pkl_data:
    # 1. load pkl data
    scene_data = pickle.load(scene_pkl_data)

    if VisMod == 1:
        # 2.select time slice
        row_num = scene_data.shape[0]
        row_idx = random.randint(0, row_num-1)
        print('Select time slice idx', row_idx, 'in', row_num)
        print(scene_data.iloc[row_idx])

        # 3. plot env info
        utils.plot_scene_pkl(scene_pkl_file_name, scene_data, row_idx, scene_pkl_file_path)

    elif VisMod == 2:
        # 2.select time slice
        row_num = scene_data.shape[0]
        row_idx = 0
        print('Select time slice idx', row_idx, 'in', row_num)
        print(scene_data.iloc[row_idx])

        # 3. plot env info
        utils.plot_scene_pkl(scene_pkl_file_name, scene_data, row_idx, scene_pkl_file_path)

    elif VisMod == 3:
        row_num = scene_data.shape[0]
        print('Num of time slice: ', row_num)
        for row_idx in range(0, row_num):
            utils.plot_scene_pkl(scene_pkl_file_name, scene_data, row_idx, scene_pkl_file_path)