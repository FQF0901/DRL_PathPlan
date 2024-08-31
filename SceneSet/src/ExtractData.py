"""
@author: fqf
@time: 20240717
@file: ExtractData.py
@description: Batch extract data(The memory usage of a time slice is about 3kb)
"""

import os
import sys
import DataUtil
import ExtractTimeSlice
import datetime
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import utils
from Util import Config

# ==========================================================
# ======================== SceneSet ========================
# ==========================================================

# ------------------------- Config -------------------------
# 1.1 mf4 location
mf4_raw_file_path = Config.StorePath.mf4_raw_file_path    # [ Config ]: The folder address where mf4 is located

# 1.2 Sampling strategy 
sample_mode = 3 # [ Config ]: 1 = Uniform Sampling besides plan time, 2 = Random Sampling besides plan time, 3 = Advanced Sampling Solutions
sample_nums = 6 # [ Config ]: 0 = Adapt according to mf4 duration, others = sample number

# 1.3 Store location
scene_slice_data_path = Config.StorePath.scene_slice_data_path # if scene_slice_data_path = [], the data will be stored in the default path
dataset_folder_path, failinfo_file_path = DataUtil.set_store_location(scene_slice_data_path)

# ------------------- Time slice sampling ------------------
# 2.1 Find all mf4 files in the folder (including subfolders)
file_list = []
file_list = DataUtil.get_file_list_from_dir(mf4_raw_file_path, file_list)

with tqdm(total=int(len(file_list)), dynamic_ncols=True, desc='Progress Bar') as pbar:
    for i, file in enumerate(file_list, start=1):
        try:
        # 2.2 Store sample time slice into .pkl
            log_info = ''
            df, log_info = ExtractTimeSlice.ets(file, sample_mode, sample_nums)
            data_file_name = DataUtil.get_store_file_name(file)

            if log_info:
                raise Exception("There seems to be a signal loss.")
            else:
                pkl_data_name = DataUtil.store_in_pkl(df, dataset_folder_path, 100) # write scene info into pkl
                # print('Extract slice suceefully ! ' + os.path.basename(file) + ' into ' + pkl_data_name)
        
        except Exception as e:
            # 2.3 Store extract fail info into log.txt
            with open(failinfo_file_path, "a") as myfile:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                myfile.write(f"{current_time} {file} {log_info}\n") # write fail info into log

            # if log_info:
            #     print(utils.HighLightRedMsg('Extract slice fail due to signal loss! See more in log file'))
            # else:
            #     print(utils.HighLightRedMsg('Extract slice fail due to unknown cause! See more in log file'))

        # 3. Progress Bar
        cycle_interval = 10
        if i % cycle_interval == 0:
                pbar.set_postfix({
                    'episode': '%d' % (i)
                    })
                
                pbar.update(cycle_interval)

print('===== Scene extraction done ! =====')
