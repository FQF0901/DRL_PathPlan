"""
@author: Fqf
@time: 20240717
@file: utils.py
@description: Shared Libraries for All users
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import Config

# ==========================================================
# ======================== Message =========================
# ==========================================================

def HighLightGreenMsg(message):
    """Set green and bold text"""
    highlighted_message = f"\033[1;32m{message}\033[0m"
    return highlighted_message

def HighLightRedMsg(message):
    """Set red and bold text"""
    highlighted_message = f"\033[1;31m{message}\033[0m"
    return highlighted_message

# ==========================================================
# ===================== Visualization ======================
# ==========================================================

def get_Veh_corners(x, y, yaw_rad, LatMargin = Config.VehPara.LatMargin, LgtMargin = Config.VehPara.LgtMargin):
    """CalcuLatMargine the four corners of the vehicle based on the rear axle center, length and VehicleWidth"""
    rear_to_front = Config.VehPara.VehicleLength / 2 + Config.VehPara.Center2RearAxle
    rear_to_back = Config.VehPara.VehicleLength / 2 - Config.VehPara.Center2RearAxle

    # 1. The local coordinates of the four corner points of the vehicle relative to the center of the vehicle's rear axle
    local_corners = np.array([
        [-rear_to_back - LgtMargin, -Config.VehPara.VehicleWidth / 2 - LatMargin],
        [-rear_to_back - LgtMargin, Config.VehPara.VehicleWidth / 2 + LatMargin],
        [rear_to_front + LgtMargin, Config.VehPara.VehicleWidth / 2 + LatMargin],
        [rear_to_front + LgtMargin, -Config.VehPara.VehicleWidth / 2 - LatMargin]
    ])

    # 2. Rotate according to the vehicle's heading angle (yaw)
    rotation_matrix = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad)],
        [np.sin(yaw_rad), np.cos(yaw_rad)]
    ])
    rotated_corners = np.dot(local_corners, rotation_matrix.T)

    # 3. Translate the four corner points of the vehicle to the global coordinate system
    global_corners = rotated_corners + np.array([x, y])

    # 4. Convert to data type
    veh_rect_x = [global_corners[0][0], global_corners[1][0], global_corners[2][0], global_corners[3][0], global_corners[0][0]]
    veh_rect_y = [global_corners[0][1], global_corners[1][1], global_corners[2][1], global_corners[3][1], global_corners[0][1]]
    veh_rect_corner = [veh_rect_x, veh_rect_y]

    return veh_rect_corner

def plot_scene_pkl(scene_pkl_file, pkl_data, row_idx, store_path=os.getcwd()):
    """Plot the env at a certain time slice in the scene pkl file"""
    # plt.interactive(False)  # 禁用交互模式

    # 1 SP
    plt.plot(pkl_data.iloc[row_idx]['StartPose'][0], 
            pkl_data.iloc[row_idx]['StartPose'][1], color='cyan', marker='o', markersize=2)
    sp_veh_rect = get_Veh_corners(pkl_data.iloc[row_idx]['StartPose'][0], 
                                        pkl_data.iloc[row_idx]['StartPose'][1], 
                                        pkl_data.iloc[row_idx]['StartPose'][2], 0, 0)
    plt.plot(sp_veh_rect[0], sp_veh_rect[1], color='cyan', linestyle='-', linewidth=1)

    # 2 TP
    plt.plot(pkl_data.iloc[row_idx]['TargetPose'][0], 
            pkl_data.iloc[row_idx]['TargetPose'][1], color='green', marker='o', markersize=2)
    tp_veh_rect = get_Veh_corners(pkl_data.iloc[row_idx]['TargetPose'][0], 
                                        pkl_data.iloc[row_idx]['TargetPose'][1], 
                                        pkl_data.iloc[row_idx]['TargetPose'][2], 0, 0)
    plt.plot(tp_veh_rect[0], tp_veh_rect[1], color='green', linestyle='-', linewidth=1)

    # 3 PSD
    plt.plot(pkl_data.iloc[row_idx]['ParkingSlot_x'], 
            pkl_data.iloc[row_idx]['ParkingSlot_y'], color='blue', linestyle='--', linewidth=1)
    
    # 4 OD
    for i in range(0, pkl_data.iloc[row_idx]['OD_Number'][0]):
        plt.plot(pkl_data.iloc[row_idx]['OD_x'][i], 
                pkl_data.iloc[row_idx]['OD_y'][i], color='magenta', linestyle='-', linewidth=1)

    # 5 FSB
    for i in range(0, pkl_data.iloc[row_idx]['FSB_Number'][0]):
        plt.plot(pkl_data.iloc[row_idx]['FSB_x'][i], 
                pkl_data.iloc[row_idx]['FSB_y'][i], color='red', linestyle='-', linewidth=1)

    # 6. PostProcess
    plt.axis([-16, 16, -9, 9])
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('X / m')
    plt.ylabel('Y / m')
    title_text = (str(pkl_data.iloc[row_idx]['FileName'])
                  + '\nTimeStamp = ' + f"{pkl_data.iloc[row_idx]['TimeStamp']:.2f}"
                  + ', PrkMod = ' + str(pkl_data.iloc[row_idx]['PrkMod'])
                  + ', SlotType = ' + str(pkl_data.iloc[row_idx]['ParkingSlot_type']))
    plt.title(title_text, fontsize=6)

    # plt.show()
    filename = f"{store_path}/{os.path.basename(scene_pkl_file).rsplit('.', 1)[0]}_rowidx{row_idx}.jpg"
    plt.savefig(filename)
    plt.close()  # Close the plot to free up memory

    # print(f"-- Plot saved as {os.path.basename(filename)} in {store_path}")