"""
@author: Yang Wu
@time: 20240717
@file: DataUtil.py
@description: Shared Libraries for Extract_Ego_Pose_from_MF4.py
"""

import math
import numpy as np
import pandas as pd
from asammdf import MDF, set_global_option  # For processing MDF files
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import utils

set_global_option("raise_on_multiple_occurrences", False)

#求斜库位可泊库位长宽
def real_slot_dimension(p0,p1,p2):
    if not ((p0 == p1) or (p1 == p2)):
        #计算P0P1边与P1P2边夹角
        line_width =  math.dist(p0, p1)
        line_length = math.dist(p1, p2)
        diagonal_length = math.dist(p0,p2)
        #余弦函数计算夹角
        angle = np.arccos((pow(line_width,2) + pow(line_length,2)-pow(diagonal_length,2))/(2*line_width*line_length))
        if (0.5*math.pi < angle):
            angle = math.pi - angle
        slot_length = round((line_length), 2)
        slot_width = round((line_width*np.sin(angle)), 2)
    else:
          slot_length = 0
          slot_width = 0
          
    return slot_length, slot_width

#将坐标系总后轴中心转到进控时刻的大地坐标系
def point_to_ground(x,y,yaw, ego_pose_x, ego_pose_y, ego_pose_yaw):
    ground_x = []
    ground_y = []
    ground_yaw = []
    delta_x = ego_pose_y
    delta_y = ego_pose_x
    delta_z = 0
    cosine_theta = math.cos(-ego_pose_yaw)
    sine_theta = math.sin(-ego_pose_yaw)

    #定义原始坐标矩阵
    Coord_Matrix = np.array([x, y, yaw, 1])

    #定义变换矩阵
    Transversion_Matrix = np.array([[cosine_theta,       -sine_theta,         0,          0],
                            [sine_theta,          cosine_theta,        0,          0],
                            [0,                   0,                   1,          0],
                            [delta_y,             delta_x,             delta_z,    1]])
    Inverse_Transversion_Matrix = np.linalg.inv(Transversion_Matrix) 
    Result_Matrix = Coord_Matrix.dot(Inverse_Transversion_Matrix)                   
    ground_x = Result_Matrix.tolist()[0]
    ground_y = Result_Matrix.tolist()[1]
    ground_yaw = yaw-ego_pose_yaw

    return ground_x, ground_y, ground_yaw


def get_slot_information(mdf_resample, index):
    slot_x = [(mdf_resample.loc[
                    index, 'TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv.PS_P0_X_ISO8855_Rear_Axle_m._0_']),
                (mdf_resample.loc[
                    index, 'TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv.PS_P1_X_ISO8855_Rear_Axle_m._0_']),
                (mdf_resample.loc[
                    index, 'TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv.PS_P2_X_ISO8855_Rear_Axle_m._0_']),
                (mdf_resample.loc[
                    index, 'TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv.PS_P3_X_ISO8855_Rear_Axle_m._0_'])
                ]
    slot_y = [(mdf_resample.loc[
                    index, 'TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv.PS_P0_Y_ISO8855_Rear_Axle_m._0_']),
                (mdf_resample.loc[
                    index, 'TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv.PS_P1_Y_ISO8855_Rear_Axle_m._0_']),
                (mdf_resample.loc[
                    index, 'TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv.PS_P2_Y_ISO8855_Rear_Axle_m._0_']),
                (mdf_resample.loc[
                    index, 'TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv.PS_P3_Y_ISO8855_Rear_Axle_m._0_'])
                ]
    slot_type = mdf_resample.loc[index, 'TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv.PS_SlotType._0_']
    slot_position = mdf_resample.loc[index, 'TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv.PS_Position._0_']

    return slot_type, slot_position, slot_x, slot_y

def get_slot_information_CPA(mdf_resample, index):
    slot_x = [(mdf_resample.loc[
                    index, 'InLy_ZF_Fusion_Parking_Slot_Set_o_obv.PS_P0_X_ISO8855_Rear_Axle_m._0_']),
                (mdf_resample.loc[
                    index, 'InLy_ZF_Fusion_Parking_Slot_Set_o_obv.PS_P1_X_ISO8855_Rear_Axle_m._0_']),
                (mdf_resample.loc[
                    index, 'InLy_ZF_Fusion_Parking_Slot_Set_o_obv.PS_P2_X_ISO8855_Rear_Axle_m._0_']),
                (mdf_resample.loc[
                    index, 'InLy_ZF_Fusion_Parking_Slot_Set_o_obv.PS_P3_X_ISO8855_Rear_Axle_m._0_']),
                (mdf_resample.loc[
                    index, 'InLy_ZF_Fusion_Parking_Slot_Set_o_obv.PS_P0_X_ISO8855_Rear_Axle_m._0_'])
                ]
    slot_y = [(mdf_resample.loc[
                    index, 'InLy_ZF_Fusion_Parking_Slot_Set_o_obv.PS_P0_Y_ISO8855_Rear_Axle_m._0_']),
                (mdf_resample.loc[
                    index, 'InLy_ZF_Fusion_Parking_Slot_Set_o_obv.PS_P1_Y_ISO8855_Rear_Axle_m._0_']),
                (mdf_resample.loc[
                    index, 'InLy_ZF_Fusion_Parking_Slot_Set_o_obv.PS_P2_Y_ISO8855_Rear_Axle_m._0_']),
                (mdf_resample.loc[
                    index, 'InLy_ZF_Fusion_Parking_Slot_Set_o_obv.PS_P3_Y_ISO8855_Rear_Axle_m._0_']),
                (mdf_resample.loc[
                    index, 'InLy_ZF_Fusion_Parking_Slot_Set_o_obv.PS_P0_Y_ISO8855_Rear_Axle_m._0_'])
                ]
    return slot_x, slot_y


def get_object_information(mdf_resample, index):
    x = []
    y = []
    object_number = []
    confidence = []

    object_number.append(mdf_resample.loc[index, 'SIFOR1_Valid_Fusion_Object_Set.Number_Of_Valid_Objects'])
    for i in range(0, len(mdf_resample.loc[index, 'InLy_ZF_Fusion_Object_Set_o_obv.Object_Point_0_X'] - 1)):
        Object_x = []
        Object_y = []
        Object_confidence = []

        Object_x.append(mdf_resample.loc[index, 'InLy_ZF_Fusion_Object_Set_o_obv.Object_Point_0_X'][i])
        Object_x.append(mdf_resample.loc[index, 'InLy_ZF_Fusion_Object_Set_o_obv.Object_Point_1_X'][i])
        Object_x.append(mdf_resample.loc[index, 'InLy_ZF_Fusion_Object_Set_o_obv.Object_Point_2_X'][i])
        Object_x.append(mdf_resample.loc[index, 'InLy_ZF_Fusion_Object_Set_o_obv.Object_Point_3_X'][i])

        Object_y.append(mdf_resample.loc[index, 'InLy_ZF_Fusion_Object_Set_o_obv.Object_Point_0_Y'][i])
        Object_y.append(mdf_resample.loc[index, 'InLy_ZF_Fusion_Object_Set_o_obv.Object_Point_1_Y'][i])
        Object_y.append(mdf_resample.loc[index, 'InLy_ZF_Fusion_Object_Set_o_obv.Object_Point_2_Y'][i])
        Object_y.append(mdf_resample.loc[index, 'InLy_ZF_Fusion_Object_Set_o_obv.Object_Point_3_Y'][i])

        Object_confidence.append(mdf_resample.loc[index, 'SIFOR1_Valid_Fusion_Object_Set.Object_Confidence'][i])

        x.append(Object_x)
        y.append(Object_y)
        confidence.append(Object_confidence)

    
    return object_number, confidence, x, y

def get_freespace_information(mdf_resample, index):
    x = []
    y = []
    FSB_Number = []
    confidence = []

    FSB_Number.append(mdf_resample.loc[index, 'SIFOR1_Valid_Free_Space_Boundary_Set.Number_Of_Valid_Objects'])
    for i in range(0, len(mdf_resample.loc[index, 'SIFOR1_Valid_Free_Space_Boundary_Set.Freespace_Boundary_Point_0_X'] - 1)):
        FSB_x = []
        FSB_y = []
        FSB_Confidence = []
        FSB_Confidence.append(mdf_resample.loc[index, 'SIFOR1_Valid_Free_Space_Boundary_Set.Freespace_Boundary_Confidence'][i])
        FSB_x.append(mdf_resample.loc[index, 'SIFOR1_Valid_Free_Space_Boundary_Set.Freespace_Boundary_Point_0_X'][i])
        FSB_x.append(mdf_resample.loc[index, 'SIFOR1_Valid_Free_Space_Boundary_Set.Freespace_Boundary_Point_1_X'][i])
        FSB_x.append(mdf_resample.loc[index, 'SIFOR1_Valid_Free_Space_Boundary_Set.Freespace_Boundary_Point_2_X'][i])
        FSB_x.append(mdf_resample.loc[index, 'SIFOR1_Valid_Free_Space_Boundary_Set.Freespace_Boundary_Point_3_X'][i])

        FSB_y.append(mdf_resample.loc[index, 'SIFOR1_Valid_Free_Space_Boundary_Set.Freespace_Boundary_Point_0_Y'][i])
        FSB_y.append(mdf_resample.loc[index, 'SIFOR1_Valid_Free_Space_Boundary_Set.Freespace_Boundary_Point_1_Y'][i])
        FSB_y.append(mdf_resample.loc[index, 'SIFOR1_Valid_Free_Space_Boundary_Set.Freespace_Boundary_Point_2_Y'][i])
        FSB_y.append(mdf_resample.loc[index, 'SIFOR1_Valid_Free_Space_Boundary_Set.Freespace_Boundary_Point_3_Y'][i])

        x.append(FSB_x)
        y.append(FSB_y)
        confidence.append(FSB_Confidence)
    return FSB_Number, confidence, x, y


def get_trajectory_information(mdf_resample, index, ego_pose_x, ego_pose_y, ego_pose_yaw):
    Trajectory_x = []
    Trajectory_y = []
    Trajectory_Yaw_Angle = []
    Vehicle_trace_x = []
    Vehicle_trace_y = []
    
    Number_Of_Elements = mdf_resample.loc[index, 'TA_ZF_Trajectory_Type_o_obv.Number_Of_Elements']
    for i in range(Number_Of_Elements[0]): 
        element_x = mdf_resample.loc[index, 'TA_ZF_Trajectory_Type_o_obv.x'][i]
        element_y = mdf_resample.loc[index, 'TA_ZF_Trajectory_Type_o_obv.y'][i]
        element_yaw_angle = mdf_resample.loc[index, 'TA_ZF_Trajectory_Type_o_obv.Yaw_Angle'][i]
        Trajectory_element_x, Trajectory_element_y, Trajectory_element_yaw_angle = point_to_ground (element_x, element_y, element_yaw_angle, ego_pose_x, ego_pose_y, ego_pose_yaw)
        Vehicle_trace__element_x, Vehicle_trace__element_y = point_to_vehicle_box_converter (Trajectory_element_x,Trajectory_element_y,Trajectory_element_yaw_angle,1,0)
        Trajectory_x.append(Trajectory_element_x)
        Trajectory_y.append(Trajectory_element_y)
        Vehicle_trace_x.append(Vehicle_trace__element_x)
        Vehicle_trace_y.append(Vehicle_trace__element_y)

        Trajectory_Yaw_Angle.append(Trajectory_element_yaw_angle)
            
    return Trajectory_x, Trajectory_y, Vehicle_trace_x, Vehicle_trace_y, Trajectory_Yaw_Angle

def get_start_pose_information(mdf_resample, index):
    StartPose = []
    StartPose.append(mdf_resample.loc[index, 'HAS_Selected_Start_Position_X_m'])
    StartPose.append(mdf_resample.loc[index, 'HAS_Selected_Start_Position_Y_m'])
    StartPose.append(mdf_resample.loc[index, 'HAS_Selected_Start_Yaw_Angle_rad'])

    return StartPose

def get_target_pose_information(mdf_resample, index):
    TargetPose = []
    TargetPose.append(mdf_resample.loc[index, 'HAS_Selected_Target_Position_X_m'])
    TargetPose.append(mdf_resample.loc[index, 'HAS_Selected_Target_Position_Y_m'])
    TargetPose.append(mdf_resample.loc[index, 'HAS_Selected_Target_Yaw_Angle_rad'])

    return TargetPose

def get_intermediate_target_position(mdf_resample, index):
    Intermediate_Target_Pose_Ego_x = mdf_resample.loc[index, 'HAS_Selected_Intermediate_Target_Position_X_m']
    Intermediate_Target_Pose_Ego_y = mdf_resample.loc[index, 'HAS_Selected_Intermediate_Target_Position_Y_m']
    Intermediate_Target_Pose_Ego_yaw = mdf_resample.loc[index, 'HAS_Selected_Intermediate_Target_Yaw_Angle_rad']
    Intermediate_Target_Pose_x,Intermediate_Target_Pose_y = point_to_vehicle_box_converter (Intermediate_Target_Pose_Ego_x,Intermediate_Target_Pose_Ego_y,Intermediate_Target_Pose_Ego_yaw,1,0)

    return Intermediate_Target_Pose_x, Intermediate_Target_Pose_y

def get_intermediate_target_position_DC(mdf_resample, index):
    Intermediate_Target_Pose_Ego_x = mdf_resample.loc[index, 'HAS_Selected_Intermediate_Target_Position_X_m[0]']
    Intermediate_Target_Pose_Ego_y = mdf_resample.loc[index, 'HAS_Selected_Intermediate_Target_Position_Y_m[0]']
    Intermediate_Target_Pose_Ego_yaw = mdf_resample.loc[index, 'HAS_Selected_Intermediate_Target_Yaw_Angle_rad[0]']
    Intermediate_Target_Pose_x,Intermediate_Target_Pose_y = point_to_vehicle_box_converter (Intermediate_Target_Pose_Ego_x,Intermediate_Target_Pose_Ego_y,Intermediate_Target_Pose_Ego_yaw,1,0)

    return Intermediate_Target_Pose_x, Intermediate_Target_Pose_y
                        
def get_intermediate_start_position(mdf_resample, index):
    Intermediate_Start_Pose_x = []
    Intermediate_Start_Pose_y = []
    Intermediate_Start_Pose_Ego_x = mdf_resample.loc[index, 'HAS_Selected_Intermediate_Start_Position_X_m']
    Intermediate_Start_Pose_Ego_y = mdf_resample.loc[index, 'HAS_Selected_Intermediate_Start_Position_Y_m']
    Intermediate_Start_Pose_Ego_yaw = mdf_resample.loc[index, 'HAS_Selected_Intermediate_Start_Yaw_Angle_rad']

    if not ( Intermediate_Start_Pose_Ego_x == 0 and Intermediate_Start_Pose_Ego_y == 0):
        Intermediate_Start_Pose_x, Intermediate_Start_Pose_y = point_to_vehicle_box_converter (Intermediate_Start_Pose_Ego_x,Intermediate_Start_Pose_Ego_y,Intermediate_Start_Pose_Ego_yaw,1,0)

    return Intermediate_Start_Pose_x, Intermediate_Start_Pose_y

#For SDF data use only

def get_slot_information_SDF(mdf_resample, index):
    slot_x = [(mdf_resample.loc[
                    index, 'libzeekr_pecu_algo_TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv_PS_P0_X_ISO8855_Rear_Axle_m'][0]),
                (mdf_resample.loc[
                    index, 'libzeekr_pecu_algo_TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv_PS_P1_X_ISO8855_Rear_Axle_m'][0]),
                (mdf_resample.loc[
                    index, 'libzeekr_pecu_algo_TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv_PS_P2_X_ISO8855_Rear_Axle_m'][0]),
                (mdf_resample.loc[
                    index, 'libzeekr_pecu_algo_TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv_PS_P3_X_ISO8855_Rear_Axle_m'][0]),
                (mdf_resample.loc[
                    index, 'libzeekr_pecu_algo_TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv_PS_P0_X_ISO8855_Rear_Axle_m'][0])
                ]
    slot_y = [(mdf_resample.loc[
                    index, 'libzeekr_pecu_algo_TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv_PS_P0_Y_ISO8855_Rear_Axle_m'][0]),
                (mdf_resample.loc[
                    index, 'libzeekr_pecu_algo_TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv_PS_P1_Y_ISO8855_Rear_Axle_m'][0]),
                (mdf_resample.loc[
                    index, 'libzeekr_pecu_algo_TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv_PS_P2_Y_ISO8855_Rear_Axle_m'][0]),
                (mdf_resample.loc[
                    index, 'libzeekr_pecu_algo_TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv_PS_P3_Y_ISO8855_Rear_Axle_m'][0]),
                (mdf_resample.loc[
                    index, 'libzeekr_pecu_algo_TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv_PS_P0_Y_ISO8855_Rear_Axle_m'][0])
                ]
    return slot_x, slot_y

def get_object_information_SDF(mdf_resample, index, i):
    Object_x = []
    Object_y = []
    Object_Type = []
    Object_Position_x = []
    Object_Position_y = []

    Object_x.append(mdf_resample.loc[index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_HI_ZF_Fusion_Object_Set_IRV_Object_Point_0_X'][i])
    Object_x.append(mdf_resample.loc[index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_HI_ZF_Fusion_Object_Set_IRV_Object_Point_1_X'][i])
    Object_x.append(mdf_resample.loc[index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_HI_ZF_Fusion_Object_Set_IRV_Object_Point_2_X'][i])
    Object_x.append(mdf_resample.loc[index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_HI_ZF_Fusion_Object_Set_IRV_Object_Point_3_X'][i])
    Object_x.append(mdf_resample.loc[index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_HI_ZF_Fusion_Object_Set_IRV_Object_Point_0_X'][i])

    Object_y.append(mdf_resample.loc[index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_HI_ZF_Fusion_Object_Set_IRV_Object_Point_0_Y'][i])
    Object_y.append(mdf_resample.loc[index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_HI_ZF_Fusion_Object_Set_IRV_Object_Point_1_Y'][i])
    Object_y.append(mdf_resample.loc[index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_HI_ZF_Fusion_Object_Set_IRV_Object_Point_2_Y'][i])
    Object_y.append(mdf_resample.loc[index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_HI_ZF_Fusion_Object_Set_IRV_Object_Point_3_Y'][i])
    Object_y.append(mdf_resample.loc[index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_HI_ZF_Fusion_Object_Set_IRV_Object_Point_0_Y'][i])

    Object_Type.append(mdf_resample.loc[index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_HI_ZF_Fusion_Object_Set_IRV_Object_Type'][i])
    Object_Position_x.append(0.5*(Object_x[0] + Object_x[1]))
    Object_Position_y.append(0.5*(Object_y[0] + Object_y[1]))

    return Object_x, Object_y, Object_Position_x, Object_Position_y, Object_Type

def get_freespace_information_SDF(mdf_resample, index, ego_pose_x, ego_pose_y, ego_pose_yaw, i):
    FSD_Bound_x = []
    FSD_Bound_y = []
    x_0 = mdf_resample.loc[index, 'libzeekr_pecu_algo_SIFOR1_Valid_Free_Space_Boundary_Set_Freespace_Boundary_Point_0_X'][i]
    x_1 = mdf_resample.loc[index, 'libzeekr_pecu_algo_SIFOR1_Valid_Free_Space_Boundary_Set_Freespace_Boundary_Point_1_X'][i]
    x_2 = mdf_resample.loc[index, 'libzeekr_pecu_algo_SIFOR1_Valid_Free_Space_Boundary_Set_Freespace_Boundary_Point_2_X'][i]
    x_3 = mdf_resample.loc[index, 'libzeekr_pecu_algo_SIFOR1_Valid_Free_Space_Boundary_Set_Freespace_Boundary_Point_3_X'][i]

    y_0 = mdf_resample.loc[index, 'libzeekr_pecu_algo_SIFOR1_Valid_Free_Space_Boundary_Set_Freespace_Boundary_Point_0_Y'][i]
    y_1 = mdf_resample.loc[index, 'libzeekr_pecu_algo_SIFOR1_Valid_Free_Space_Boundary_Set_Freespace_Boundary_Point_1_Y'][i]
    y_2 = mdf_resample.loc[index, 'libzeekr_pecu_algo_SIFOR1_Valid_Free_Space_Boundary_Set_Freespace_Boundary_Point_1_Y'][i]
    y_3 = mdf_resample.loc[index, 'libzeekr_pecu_algo_SIFOR1_Valid_Free_Space_Boundary_Set_Freespace_Boundary_Point_1_Y'][i]

    FSD_Bound_x.append(x_0)
    FSD_Bound_x.append(x_1)
    FSD_Bound_x.append(x_2)
    FSD_Bound_x.append(x_3)
    FSD_Bound_y.append(y_0)
    FSD_Bound_y.append(y_1)
    FSD_Bound_y.append(y_2)
    FSD_Bound_y.append(y_3)
    
    return FSD_Bound_x, FSD_Bound_y

def get_trajectory_information_SDF(mdf_resample, index, ego_pose_x, ego_pose_y, ego_pose_yaw):
    Trajectory_x = []
    Trajectory_y = []
    Trajectory_Yaw_Angle = []
    Vehicle_trace_x = []
    Vehicle_trace_y = []
    
    Number_Of_Elements = mdf_resample.loc[index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_ZF_Trajectory_Type_IRV_Number_Of_Elements']
    for i in range(Number_Of_Elements): 
        element_x = mdf_resample.loc[index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_ZF_Trajectory_Type_IRV_x'][i]
        element_y = mdf_resample.loc[index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_ZF_Trajectory_Type_IRV_y'][i]
        element_yaw_angle = mdf_resample.loc[index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_ZF_Trajectory_Type_IRV_Yaw_Angle'][i]
        Trajectory_element_x, Trajectory_element_y, Trajectory_element_yaw_angle = point_to_ground (element_x, element_y, element_yaw_angle, ego_pose_x, ego_pose_y, ego_pose_yaw)
        Vehicle_trace__element_x, Vehicle_trace__element_y = point_to_vehicle_box_converter (element_x,element_y,element_yaw_angle,1,0)
        Trajectory_x.append(element_x)
        Trajectory_y.append(element_y)
        Vehicle_trace_x.append(Vehicle_trace__element_x)
        Vehicle_trace_y.append(Vehicle_trace__element_y)

        Trajectory_Yaw_Angle.append(element_yaw_angle)


            
    return Trajectory_x, Trajectory_y, Vehicle_trace_x, Vehicle_trace_y

def get_target_pose_information_SDF(mdf_resample, index):
    Target_Pose_Ego_x = mdf_resample.loc[index, 'libzeekr_pecu_algo_HAS_Selected_Target_Position_X_m']
    Target_Pose_Ego_y = mdf_resample.loc[index, 'libzeekr_pecu_algo_HAS_Selected_Target_Position_Y_m']
    Target_Pose_Ego_yaw = mdf_resample.loc[index, 'libzeekr_pecu_algo_HAS_Selected_Target_Yaw_Angle_rad']
    Target_Pose_x, Target_Pose_y = point_to_vehicle_box_converter (Target_Pose_Ego_x,Target_Pose_Ego_y,Target_Pose_Ego_yaw,0,0)

    return Target_Pose_x, Target_Pose_y, Target_Pose_Ego_yaw

def get_intermediate_target_position_SDF(mdf_resample, index):
    Intermediate_Target_Pose_Ego_x = mdf_resample.loc[index, 'libzeekr_pecu_algo_HAS_Selected_Intermediate_Target_Position_X_m']
    Intermediate_Target_Pose_Ego_y = mdf_resample.loc[index, 'libzeekr_pecu_algo_HAS_Selected_Intermediate_Target_Position_Y_m']
    Intermediate_Target_Pose_Ego_yaw = mdf_resample.loc[index, 'libzeekr_pecu_algo_HAS_Selected_Intermediate_Target_Yaw_Angle_rad']
    Intermediate_Target_Pose_x,Intermediate_Target_Pose_y = point_to_vehicle_box_converter (Intermediate_Target_Pose_Ego_x,Intermediate_Target_Pose_Ego_y,Intermediate_Target_Pose_Ego_yaw,1,0)

    return Intermediate_Target_Pose_x, Intermediate_Target_Pose_y

def get_intermediate_start_position_SDF(mdf_resample, index):
    Intermediate_Start_Pose_x = []
    Intermediate_Start_Pose_y = []
    Intermediate_Start_Pose_Ego_x = mdf_resample.loc[index, 'libzeekr_pecu_algo_HAS_Selected_Intermediate_Start_Position_X_m']
    Intermediate_Start_Pose_Ego_y = mdf_resample.loc[index, 'libzeekr_pecu_algo_HAS_Selected_Intermediate_Start_Position_Y_m']
    Intermediate_Start_Pose_Ego_yaw = mdf_resample.loc[index, 'libzeekr_pecu_algo_HAS_Selected_Intermediate_Start_Yaw_Angle_rad']

    if not ( Intermediate_Start_Pose_Ego_x == 0 and Intermediate_Start_Pose_Ego_y == 0):
        Intermediate_Start_Pose_x, Intermediate_Start_Pose_y = point_to_vehicle_box_converter (Intermediate_Start_Pose_Ego_x,Intermediate_Start_Pose_Ego_y,Intermediate_Start_Pose_Ego_yaw,1,0)

    return Intermediate_Start_Pose_x, Intermediate_Start_Pose_y

def get_virtual_boundary_information_SDF(mdf_resample, index, i):
    Virtual_Boundary_x = []
    Virtual_Boundary_y = []
    Virtual_Boundary_x.append(mdf_resample.loc[
                        index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_ZF_Virtual_Boundary_Set_IRV_Object_Point_0_X'][i])
    Virtual_Boundary_x.append(mdf_resample.loc[
                        index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_ZF_Virtual_Boundary_Set_IRV_Object_Point_1_X'][i])
    Virtual_Boundary_x.append(mdf_resample.loc[
                        index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_ZF_Virtual_Boundary_Set_IRV_Object_Point_2_X'][i])
    Virtual_Boundary_x.append(mdf_resample.loc[
                        index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_ZF_Virtual_Boundary_Set_IRV_Object_Point_3_X'][i])
    Virtual_Boundary_x.append(mdf_resample.loc[
                        index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_ZF_Virtual_Boundary_Set_IRV_Object_Point_0_X'][i])
    Virtual_Boundary_y.append(mdf_resample.loc[
                        index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_ZF_Virtual_Boundary_Set_IRV_Object_Point_0_Y'][i])
    Virtual_Boundary_y.append(mdf_resample.loc[
                        index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_ZF_Virtual_Boundary_Set_IRV_Object_Point_1_Y'][i])
    Virtual_Boundary_y.append(mdf_resample.loc[
                        index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_ZF_Virtual_Boundary_Set_IRV_Object_Point_2_Y'][i])
    Virtual_Boundary_y.append(mdf_resample.loc[
                        index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_ZF_Virtual_Boundary_Set_IRV_Object_Point_3_Y'][i])  
    Virtual_Boundary_y.append(mdf_resample.loc[
                        index, 'libzeekr_pecu_algo_Rte_Irv_SWC_APA_GLY_B1XE_ZF_Virtual_Boundary_Set_IRV_Object_Point_0_Y'][i])  
        
    return Virtual_Boundary_x, Virtual_Boundary_y

#后轴中心到指定坐标系转换
def rear_axil_sys_to_slot (target_x, target_y, target_yaw_angle, original_x, original_y, original_yaw):
    rear_axil_x = []
    rear_axil_y = []
    rear_axil_yaw = []
    #reserve target_z    
    target_z = 0

    delta_x = original_x - target_x
    delta_y = original_y - target_y
    delta_z = 0 - target_z
    theta = 0.5*math.pi - target_yaw_angle

    cosine_theta = math.cos(theta)
    sine_theta = math.sin(theta)

    #定义平移矩阵
    Translation_Matrix = np.array([delta_x, delta_y, delta_z])
    #定义旋转矩阵
    Transversion_Matrix = np.array([
                            [cosine_theta,        sine_theta,          0],
                            [-sine_theta,         cosine_theta,        0],
                            [0,                   0,                   1],
                            ])
    
    Result_Matrix = Translation_Matrix.dot(Transversion_Matrix)
    rear_axil_x = Result_Matrix.tolist()[0]
    rear_axil_y = Result_Matrix.tolist()[1]
    rear_axil_yaw = original_yaw
    
    return rear_axil_x, rear_axil_y, rear_axil_yaw

#笛卡尔坐标系转极坐标系
def Cartesian_to_Polar_convert (x, y):
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y,x)

    return r, theta

def export_txt(txt_path, list_name):
    list_element=open(txt_path,'w')
    for line in list_name:
        list_element.write(line+'\n')
    list_element.close()

def set_store_location(Store_path):
    failinfo_file_name = "log_FailExtractMf4.txt"

    if not os.path.exists(Store_path):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        store_folder_path = root_dir + r'\01_DataSetGen\DataSet'

        failinfo_file_path = os.path.join(store_folder_path, failinfo_file_name)
    else:
        store_folder_path = Store_path
        failinfo_file_path = os.path.join(store_folder_path, failinfo_file_name)

    print(utils.HighLightGreenMsg('Data_store_path is : ' + store_folder_path))

    return store_folder_path, failinfo_file_path

def get_store_file_name(file):
    file_name_with_extension = os.path.basename(file)
    file_name_without_extension = file_name_with_extension.rsplit('.', 1)[0]
    
    return file_name_without_extension

def get_file_list_from_dir(root_path, file_list):
    for file in os.listdir(root_path):
        full_path = os.path.join(root_path, file) 
        if os.path.isdir(full_path):
            get_file_list_from_dir(full_path, file_list)
        elif '_BEV' not in file and '_TDA4DDS' not in file and '_TDA4GMSL' not in file and file.endswith('.mf4'):
            file_list.append(full_path)

    return file_list

def store_in_pkl(new_df, store_folder_path, max_file_size_M):

    pkl_file_path = os.path.join(store_folder_path, 'mf4_time_slice_data.pkl')

    if os.path.exists(pkl_file_path):
        existing_df = pd.read_pickle(pkl_file_path)
        sum_sf = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        sum_sf = new_df
        
    if sum_sf.memory_usage(deep=True).sum() > max_file_size_M * 1024 * 1024:
        pkl_file_path = pkl_file_path.replace(".pkl", f"_part{len(os.listdir(os.path.dirname(pkl_file_path))) + 1}.pkl")
        new_df.to_pickle(pkl_file_path)
    else:
        sum_sf.to_pickle(pkl_file_path)

    return os.path.basename(pkl_file_path)