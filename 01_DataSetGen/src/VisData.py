"""
@author: fqf
@time: 20240722
@file: VisData.py
@description: Visualization for mf4 data
"""

import csv
import matplotlib.pyplot as plt
import pickle

# ==========================================================
# ===================== visualization ======================
# ==========================================================

#  -------------------------- csv --------------------------
# csv_data = r'D:\01_Proj\09_TrainData\Zeekr_PR62383_DC1E_746_R07B05DrivingAR_2024_07_20_164051#CANape_log_017.csv'

# with open(csv_data, newline='') as csvfile:
#     csvreader = csv.reader(csvfile, delimiter=',')

#     print(csvreader)

#  ------------------------- pickle -------------------------
pkl_data = r'D:\01_Proj\09_TrainData\Zeekr_PR62383_DC1E_746_R07B05DrivingAR_2024_07_20_162712#CANape_log_012.pkl'

with open(pkl_data, 'rb') as f:
    data = pickle.load(f)
    print(data)
    print('==================')
    row_idx = 2
    print(data.iloc[row_idx]['FSB_x'])
    print('==================')
    print(data.shape[0])