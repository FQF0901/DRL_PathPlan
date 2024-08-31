import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import Config

pkl_files = [f for f in os.listdir(Config.StorePath.scene_slice_data_path) if f.endswith(".pkl")]
pkl_files.sort(key=lambda x: int(x.split("_part")[1].split(".pkl")[0]) if "_part" in x else 0)

largest_suffix_pkl = os.path.join(Config.StorePath.scene_slice_data_path, pkl_files[-1])

with open(largest_suffix_pkl, 'rb') as scene_pkl_data:   #  Select scene time slice randomly
        scene_data = pickle.load(scene_pkl_data)
        row_num = scene_data.shape[0]
        
        file_time_zip = zip(scene_data['FileName'], scene_data['TimeStamp'])
        file_time_list = list(file_time_zip)
        for idx in range(0, row_num):
            print(file_time_list[idx])

        print('Scenes Num: ', row_num)