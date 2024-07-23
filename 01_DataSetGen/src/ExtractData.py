"""
@author: fqf
@time: 20240717
@file: ExtractData.py
@description: Batch extract data
"""

import os
import sys
import DataUtil
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import utils
import ExtractTimeSlice

# ==========================================================
# ======================= DataSetGen =======================
# ==========================================================

# ------------------------- Config -------------------------
# 1.1 mf4 location
mf4_folder_path = r'C:\01_Project\05_Geely\06_Test\VT_Data'    # [ Config ]: The folder address where mf4 is located

# 1.2 Sampling strategy 
sample_mode = 2 # [ Config ]: 1 = Uniform Sampling besides plan time, 2 = Random Sampling besides plan time
sample_nums = 5 # [ Config ]: 0 = Adapt according to mf4 duration, others = sample number

# 1.3 Store location
Store_path = r'D:\01_Proj\09_TrainData' # if Store_path = [], the data will be stored in the default path
dataset_folder_path, failinfo_file_path = DataUtil.set_store_location(Store_path = '')

# ------------------- Time slice sampling ------------------
# 2.1 Find all mf4 files in the folder (including subfolders)
file_list = []
file_list = DataUtil.get_file_list_from_dir(mf4_folder_path, file_list)

for i, file in enumerate(file_list, start=1):
    # try:
        # 2.2 Store sample time slice into .csv/.pkl
    df = ExtractTimeSlice.ets(file, sample_mode, sample_nums)
    data_file_name = DataUtil.get_store_file_name(file)
        # df.to_csv(os.path.join(dataset_folder_path, str(data_file_name + '.csv')), index = False, header = True)
        # df.to_pickle(os.path.join(dataset_folder_path, str(data_file_name + '.pkl')))
    DataUtil.store_in_pkl(df, dataset_folder_path, 0.1024)

    print('Extract slice suceefully ! ' + file)
    
    # except:
    #     # 2.3 Store extract fail info into log.txt
    #     with open(failinfo_file_path, "a") as myfile:
    #         myfile.write(f"{file}\n")

    #     print(utils.HighLightRedMsg('Extract slice fail ! Skip this mf4 : ' + file))