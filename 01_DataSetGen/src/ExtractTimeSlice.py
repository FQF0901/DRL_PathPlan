"""
@author: Yang Wu
@time: 20240717
@file: ExtractTimeSlice.py
@description: Perform time slice sampling on mf4 to generate a data set
"""

import pandas as pd
import numpy as np
from asammdf import MDF, set_global_option  # For processing MDF files
import os
import DataUtil 
import sys
import SignalList
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import utils

set_global_option("raise_on_multiple_occurrences", False)
folder_path = r'C:\01_Project\05_Geely\06_Test\EF1E'
os.chdir(folder_path)
files = os.listdir(folder_path)

##################################################################################################################################################################        
# Control_Panel
index_step = 500

##################################################################################################################################################################  
def ets(file, sample_mode, sample_nums):    

    df = pd.DataFrame()

    mdffile = MDF(file)
    mdf_resample = mdffile.to_dataframe(
    channels = SignalList.Signal_Library_DRL,
    raster='HMIC_HMI_OutInputs_o_obv.DrvrAsscSysDispHMI[0]', 
    use_interpolation=True, ignore_value2text_conversions=True)
    mdf_resample = mdf_resample.reset_index()

    # --------------------- Gen sample time --------------------
    # 1.1 start index
    Start_Time_Index = 1e9

    if 1 in mdf_resample['InLy_ASDMSafetyCANFD4Fr02_i_obv.DrvrAsscSysBtnPush'].values:
        Start_Time_Index = mdf_resample.loc[mdf_resample['InLy_ASDMSafetyCANFD4Fr02_i_obv.DrvrAsscSysBtnPush'] == 1].index[0]
    elif 2 in mdf_resample['InLy_ASDMSafetyCANFD4Fr02_i_obv.DrvrAsscSysBtnPush'].values:
        Start_Time_Index = min(mdf_resample.loc[mdf_resample['InLy_ASDMSafetyCANFD4Fr02_i_obv.DrvrAsscSysBtnPush'] == 2].index[0], Start_Time_Index)
    elif 8 in mdf_resample['InLy_ASDMSafetyCANFD4Fr02_i_obv.DrvrAsscSysBtnPush'].values:
        Start_Time_Index = min(mdf_resample.loc[mdf_resample['InLy_ASDMSafetyCANFD4Fr02_i_obv.DrvrAsscSysBtnPush'] == 8].index[0], Start_Time_Index)
    elif 10 in mdf_resample['InLy_ASDMSafetyCANFD4Fr02_i_obv.DrvrAsscSysBtnPush'].values:
        Start_Time_Index = min(mdf_resample.loc[mdf_resample['InLy_ASDMSafetyCANFD4Fr02_i_obv.DrvrAsscSysBtnPush'] == 10].index[0], Start_Time_Index)
    else: 
        Start_Time_Index = 1

    # 1.2 end index                   
    if 22 in mdf_resample['PFSM_SYSM_HMIOutOutputs_o_obv.PAS_State'].values:
        End_Time_Index = mdf_resample.loc[mdf_resample['PFSM_SYSM_HMIOutOutputs_o_obv.PAS_State'] == 22].index[0]
    else:
        if 30 in mdf_resample['PFSM_SYSM_HMIOutOutputs_o_obv.PAS_State'].values and (mdf_resample.loc[mdf_resample['PFSM_SYSM_HMIOutOutputs_o_obv.PAS_State'] == 30].index[0] + 20) < len(mdf_resample.index):
            End_Time_Index = mdf_resample.loc[mdf_resample['PFSM_SYSM_HMIOutOutputs_o_obv.PAS_State'] == 30].index[0] + 20
        else:
            End_Time_Index = len(mdf_resample.index) - 1

    if (30 > (End_Time_Index - Start_Time_Index)):
        index_step = 1

    # 1.3 Sample time slice
    if sample_mode == 1:    # Uniform Sampling
        time_slice_list = np.linspace(Start_Time_Index, End_Time_Index, sample_nums, endpoint=True, dtype=int)
    elif sample_mode == 2:  # Random Sampling
        time_slice_list = np.random.randint(Start_Time_Index, End_Time_Index + 1, size=sample_nums)
    else:
        time_slice_list = []

    # ------------------ Sampling key signals ------------------
    for index in time_slice_list:

        TimeStamp = mdf_resample.loc[index, 'timestamps']
        PfsmState = mdf_resample.loc[index, 'PFSM_ZF_Parking_State_o_obv']
        SsmState = mdf_resample.loc[index, 'SA_Replanning_Trigger_Type_NU_o_obv']
        Compute_State = mdf_resample.loc[index, 'PFSM_ZF_Trajectory_Compute_State_i_obv']
        Replanning_type = mdf_resample.loc[index, 'SA_Replanning_Trigger_Type_NU_o_obv']
        PrkMod = mdf_resample.loc[index, 'PFSM_SYSM_HMIOutOutputs_o_obv.ParkingSelectMode']
        StartPose = DataUtil.get_start_pose_information(mdf_resample, index)
        TargetPose = DataUtil.get_target_pose_information(mdf_resample, index)
        ParkingSlot_type, ParkingSlot_Position, ParkingSlot_x, ParkingSlot_y = DataUtil.get_slot_information(mdf_resample, index)
        Object_Number, Object_Confidence, Object_x, Object_y = DataUtil.get_object_information(mdf_resample, index)
        FSB_Number, FSB_Confidence, FSB_x, FSB_y = DataUtil.get_freespace_information(mdf_resample, index)

        new_data = pd.DataFrame({"TimeStamp": [TimeStamp],
                                "PfsmState": [PfsmState],
                                "SsmState": [SsmState],
                                "Compute_State": [Compute_State],
                                "Replanning_type": [Replanning_type],
                                "PrkMod": [PrkMod],
                                "StartPose": [StartPose],
                                "TargetPose": [TargetPose],
                                "ParkingSlot_type": [ParkingSlot_type],
                                "ParkingSlot_Position": [ParkingSlot_Position],
                                "ParkingSlot_x": [ParkingSlot_x],
                                "ParkingSlot_y": [ParkingSlot_y],
                                "Object_Number": [Object_Number],
                                "Object_Confidence": [Object_Confidence],
                                "Object_x": [Object_x],
                                "Object_y": [Object_y],
                                "FSB_Number": [FSB_Number],
                                "FSB_Confidence": [FSB_Confidence],
                                "FSB_x": [FSB_x],
                                "FSB_y": [FSB_y]
                                })
        df = pd.concat([df, new_data])

    return df
