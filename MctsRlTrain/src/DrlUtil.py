"""
@author: Fqf
@time: 20240618
@file: DrlUtil.py
@description: Shared Libraries for DRL
"""

import uuid
import csv
import math
import DrlCfg
import GlbVar
import RS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from Collection import mcts_value_net
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import Config
from Util import utils

# ==========================================================
# ===================== Bicycle Model ======================
# ==========================================================

"""Predicting ego vehicle pose using bicycle model"""
def PrdtVehPoseBicyMod(crnt_node, StrAngRate, gear, dist):

    SteerAng_rad = math.radians(Config.VehPara.MaxFrntStrAng_deg) * StrAngRate    # steer ranges form -1.f to 1.f
    Dist = dist * gear  # gear = -1/1
    
    if SteerAng_rad != 0:
        Turning_Angle_rad = math.tan(SteerAng_rad) * Dist / Config.VehPara.WheelRadius
        Turn_Radius = Dist / Turning_Angle_rad
        NextYaw_rad = math.fmod(crnt_node.yaw_rad + Turning_Angle_rad + math.pi, 2*math.pi) - math.pi
        NextX = crnt_node.x - Turn_Radius * math.sin(crnt_node.yaw_rad) + Turn_Radius * math.sin(NextYaw_rad)
        NextY = crnt_node.y + Turn_Radius * math.cos(crnt_node.yaw_rad) - Turn_Radius * math.cos(NextYaw_rad)

    else:
        Turning_Angle_rad = 0
        Turn_Radius = dist
        NextX = crnt_node.x + Dist * math.cos(crnt_node.yaw_rad)
        NextY = crnt_node.y + Dist * math.sin(crnt_node.yaw_rad)
        NextYaw_rad = crnt_node.yaw_rad

    next_node = GlbVar.Node(NextX, NextY, NextYaw_rad)
    return next_node


# ==========================================================
# ==================== Collision Check =====================
# ==========================================================

"""Calculate the four corners of the vehicle based on the rear axle center"""
def GetVehRect(crnt_node, LatMargin = Config.VehPara.LatMargin, LgtMargin = Config.VehPara.LgtMargin):
    Rear2BackDist = Config.VehPara.VehicleLength / 2 - (Config.VehPara.Center2RearAxle)
    Rear2FrtDistDist = Config.VehPara.VehicleLength / 2 + (Config.VehPara.Center2RearAxle)

    local_corners = np.array([
        [-Rear2BackDist - LgtMargin, -Config.VehPara.VehicleWidth / 2 - LatMargin],
        [-Rear2BackDist - LgtMargin, Config.VehPara.VehicleWidth / 2 + LatMargin],
        [Rear2FrtDistDist + LgtMargin, Config.VehPara.VehicleWidth / 2 + LatMargin],
        [Rear2FrtDistDist + LgtMargin, -Config.VehPara.VehicleWidth / 2 - LatMargin]
    ])

    rotation_matrix = np.array([
        [np.cos(crnt_node.yaw_rad), -np.sin(crnt_node.yaw_rad)],
        [np.sin(crnt_node.yaw_rad), np.cos(crnt_node.yaw_rad)]])
    
    rotated_corners = np.dot(local_corners, rotation_matrix.T)
    global_corners = rotated_corners + np.array([crnt_node.x, crnt_node.y])

    return global_corners

"""Calculates the projection range on the specified axis"""
def GetProjection(corners, axis):
    projections = [np.dot(corner, axis) for corner in corners]
    return min(projections), max(projections)

"""Check if two projections overlap"""
def ChkOvlp(proj1, proj2):
    return not (proj1[1] < proj2[0] or proj2[1] < proj1[0])

"""Separation axis"""
def IsOvlpSprtAxis(rect1, rect2):
    axes = []
    for rect in [rect1, rect2]:
        for i in range(len(rect)):
            edge = rect[i] - rect[i-1]
            normal = np.array([-edge[1], edge[0]])
            if not np.linalg.norm(normal) == 0: # used for FSB(line not rect)
                axes.append(normal / np.linalg.norm(normal))
    
    for axis in axes:
        proj1 = GetProjection(rect1, axis)
        proj2 = GetProjection(rect2, axis)
        if not ChkOvlp(proj1, proj2):
            return False
    return True

"""Check Collision between Veh node and all obstacles (OD / FSD)"""
def IsOvlpAllObst(crnt_node, LatMargin = Config.VehPara.LatMargin, LgtMargin = Config.VehPara.LgtMargin):
    VehRect = GetVehRect(crnt_node, LatMargin, LgtMargin)

    for ObstRect in GlbVar.PcptInfo.Obstcle_list:
        if IsOvlpSprtAxis(ObstRect, VehRect):
            return True # Overlap
    return False    # No Overlap


# ==========================================================
# ====================== reeds_shepp =======================
# ==========================================================
"""Calculate RS from start_node to goal_node"""
def CalValidRS(start_node, goal_node):  # Only RS is returned, and the bidirectionally extended A* requires additional discrete trajectories
    # 1. Calculate RS from start_node to goal_node
    RsPath = RS.calc_optimal_path(start_node, goal_node)

    # 2. Collision check, return false if there is a collision
    for i in range(0, len(RsPath.x)):
        RSnode = GlbVar.Node(RsPath.x[i], RsPath.y[i], RsPath.yaw[i])
        if IsOvlpAllObst(RSnode):
            return False, []
        
        # 3. All RS nodes are overlap-free, return true, Astar and RS path
        elif i == len(RsPath.x) - 1:
            # AstarPath = []
            # while crnt_tree_node:
            #     AstarPath.append(crnt_tree_node)
            #     crnt_tree_node = crnt_tree_node.parent  # Backtrack all tree nodes

            # return True, AstarPath[::-1], RsPath
            return True, RsPath


# ==========================================================
# ====================== Search Tree =======================
# ==========================================================

"""Expand the child nodes based on current node"""
def Expd_Node(crnt_node):
    ExpdNode_list = []
    ExpdAct_list = []
    
    for gear in DrlCfg.TreePara.GearAct_list:
        for StrAngRate in DrlCfg.TreePara.StrAct_list:
            for dist in DrlCfg.TreePara.ExpdStepLen_list:
                new_node = PrdtVehPoseBicyMod(crnt_node, StrAngRate, gear, dist)
                ExpdNode_list.append(new_node)
                ExpdAct_list.append([StrAngRate, gear, dist])

    return ExpdNode_list, ExpdAct_list

"""Check if it is a leaf node, that is, a node that has not been expanded"""
def IsLeafNode(node):
    return node.children == []


# ==========================================================
# ===================== Cost & Reward ======================
# ==========================================================

"""Check Parent state and child state is repeat or not"""
def ChkRepeatAct(node):
    SterOppoFlag = True if node.parent.ActInfo[0] == -node.ActInfo[0] else False   # Steering wheel rate complementary
    GearOppoFlag = True if node.parent.ActInfo[1] == -node.ActInfo[1] else False    # Gear position difference
    DistEqualFlag = True if node.parent.ActInfo[2] == node.ActInfo[2] else False    # Equal expansion distance
    
    return SterOppoFlag and GearOppoFlag and DistEqualFlag

"""Trajectory Value Judgment after collision free"""
def LeafNodeValueJudge(node):
    TreeLevelCost = node.TreeLvl * DrlCfg.TreePara.TreeLvlCostCoff
    ActCost = node.ActCost
    Close2ObstcCost = 0 # need to update [important]
    PathFndReward = 1 if node.type == 4 else (node.Value * 0.95)   # [important]

    TotalValue = PathFndReward  # TreeLevelCost + ActCost + Close2ObstcCost + PathFndReward # need to update [important]

    return TotalValue

def get_file_list_from_dir(root_path, file_list):
    for file in os.listdir(root_path):
        full_path = os.path.join(root_path, file) 
        if os.path.isdir(full_path):
            get_file_list_from_dir(full_path, file_list)
        elif 'mf4_time_slice_data' in file and file.endswith('.pkl'):
            file_list.append(full_path)

    return file_list

# ==========================================================
# ======================= Collection =======================
# ==========================================================
def init_mcts_info(MT, scene):
    ''' Initialize the MCTS tree '''
    if scene['PrkMod'] == 1:
        MT.RootMctsNode.node = GlbVar.Node(scene['TargetPose'][0], scene['TargetPose'][1], scene['TargetPose'][2])
        MT.GoalNode = GlbVar.Node(scene['StartPose'][0], scene['StartPose'][1], scene['StartPose'][2])
    else:
        MT.RootMctsNode.node = GlbVar.Node(scene['StartPose'][0], scene['StartPose'][1], scene['StartPose'][2])
        MT.GoalNode = GlbVar.Node(scene['TargetPose'][0], scene['TargetPose'][1], scene['TargetPose'][2])

def init_mctsefct_info(MT, scene):
    MT.RootMctsEfctNode.node = GlbVar.Node(scene['TargetPose'][0], scene['TargetPose'][1], scene['TargetPose'][2])
    MT.TargetPose = GlbVar.Node(scene['StartPose'][0], scene['StartPose'][1], scene['StartPose'][2])

def init_PcptGeo_info(scene): 
    ''' Initialize environment information '''
    GlbVar.PcptInfo = GlbVar.PcptGeo([], [], [], [])

    # SP & TP
    GlbVar.PcptInfo.StartPoint = GlbVar.Node(scene['StartPose'][0], scene['StartPose'][1], scene['StartPose'][2])
    GlbVar.PcptInfo.TargetPoint = GlbVar.Node(scene['TargetPose'][0], scene['TargetPose'][1], scene['TargetPose'][2])

    # slot
    GlbVar.PcptInfo.Slot.append(np.array([
                                [scene['ParkingSlot_x'][0], scene['ParkingSlot_y'][0]],
                                [scene['ParkingSlot_x'][1], scene['ParkingSlot_y'][1]],
                                [scene['ParkingSlot_x'][2], scene['ParkingSlot_y'][2]],
                                [scene['ParkingSlot_x'][3], scene['ParkingSlot_y'][3]]
                                ]))

    # OD and FSB
    for i in range(0, scene['OD_Number'][0]):
        GlbVar.PcptInfo.Obstcle_list.append(np.array([
                                            [scene['OD_x'][i][0], scene['OD_y'][i][0]],
                                            [scene['OD_x'][i][1], scene['OD_y'][i][1]],
                                            [scene['OD_x'][i][2], scene['OD_y'][i][2]],
                                            [scene['OD_x'][i][3], scene['OD_y'][i][3]]
                                            ]))

    for i in range(0, scene['FSB_Number'][0]):
        GlbVar.PcptInfo.Obstcle_list.append(np.array([
                                            [scene['FSB_x'][i][0], scene['FSB_y'][i][0]],
                                            [scene['FSB_x'][i][1], scene['FSB_y'][i][1]],
                                            [scene['FSB_x'][i][2], scene['FSB_y'][i][2]],
                                            [scene['FSB_x'][i][3], scene['FSB_y'][i][3]]
                                            ]))
        
def NodeNetValue():
    img = 1
    # value = mcts_value_net.policy_value_single(img)

    # return value   # Dimensions: 6 or more

# ==========================================================
# ===================== visualization ======================
# ==========================================================
#      
def plot_PcptGeo(scene_pkl_file, row_idx, store_path):
    ''' Plot PcptGeo env '''
    figsize = (384 / 100, 224 / 100)    # figsize is in inches, 384x224 pixel image at 100 dpi
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # Start Pose and Target Pose
    plt.plot(GlbVar.PcptInfo.StartPoint.x, GlbVar.PcptInfo.StartPoint.y, color='cyan', marker='o', markersize=0.5)
    sp_veh_rect = utils.get_Veh_corners(GlbVar.PcptInfo.StartPoint.x, GlbVar.PcptInfo.StartPoint.y, GlbVar.PcptInfo.StartPoint.yaw_rad, 0, 0)
    plt.plot(sp_veh_rect[0], sp_veh_rect[1], color='cyan', linestyle='-', linewidth=0.5)

    plt.plot(GlbVar.PcptInfo.TargetPoint.x, GlbVar.PcptInfo.TargetPoint.y, color='green', marker='o', markersize=0.5)
    tp_veh_rect = utils.get_Veh_corners(GlbVar.PcptInfo.TargetPoint.x, GlbVar.PcptInfo.TargetPoint.y, GlbVar.PcptInfo.TargetPoint.yaw_rad, 0, 0)
    plt.plot(tp_veh_rect[0], tp_veh_rect[1], color='green', linestyle='-', linewidth=0.5)

    # Slot
    if GlbVar.PcptInfo.Slot is not None:
        slot_polygon = patches.Polygon(GlbVar.PcptInfo.Slot[0], closed=True, edgecolor='b', facecolor='none', linestyle='--', linewidth=0.5)
        ax.add_patch(slot_polygon)

    # OD and FSB
    for rect in GlbVar.PcptInfo.Obstcle_list:
        obst_polygon = patches.Polygon(rect, closed=True, edgecolor='r', facecolor='none', linestyle='-', linewidth=0.5)
        ax.add_patch(obst_polygon)

    # Post process
    physics_size = 14
    xlim = (-physics_size, physics_size)
    ylim = (-physics_size/384*224, physics_size/384*224)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)   # Make the content fill the entire picture

    # Save fig
    filename = f"{store_path}/{os.path.basename(scene_pkl_file).rsplit('.', 1)[0]}_rowidx{row_idx}_PcptGeo.png"
    plt.savefig(filename)
    plt.close()


def GenImgLabel(scene, Img_Label_path):
    Img = scene[0]
    Label = scene[1]  

    ''' Img '''
    figsize = (384 / 100, 224 / 100)    # figsize is in inches, 384x224 pixel image at 100 dpi
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # Start Pose and Target Pose
    plt.plot(Img[0].x, Img[0].y, color='cyan', marker='o', markersize=0.5)
    sp_veh_rect = utils.get_Veh_corners(Img[0].x, Img[0].y, Img[0].yaw_rad, 0, 0)
    plt.plot(sp_veh_rect[0], sp_veh_rect[1], color='cyan', linestyle='-', linewidth=0.5)

    plt.plot(Img[1].x, Img[1].y, color='green', marker='o', markersize=0.5)
    tp_veh_rect = utils.get_Veh_corners(Img[1].x, Img[1].y, Img[1].yaw_rad, 0, 0)
    plt.plot(tp_veh_rect[0], tp_veh_rect[1], color='green', linestyle='-', linewidth=0.5)

    # Slot
    if Img[2].Slot is not None:
        slot_polygon = patches.Polygon(Img[2].Slot[0], closed=True, edgecolor='b', facecolor='none', linestyle='--', linewidth=0.5)
        ax.add_patch(slot_polygon)

    # OD and FSB
    for rect in Img[2].Obstcle_list:
        obst_polygon = patches.Polygon(rect, closed=True, edgecolor='r', facecolor='none', linestyle='-', linewidth=0.5)
        ax.add_patch(obst_polygon)

    # Post process
    physics_size = 14
    xlim = (-physics_size, physics_size)
    ylim = (-physics_size/384*224, physics_size/384*224)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)   # Make the content fill the entire picture

    # Save fig
    # filename = f"{store_path}/{os.path.basename(scene_pkl_file).rsplit('.', 1)[0]}_rowidx{row_idx}_PcptGeo.png"
    file_name = str(uuid.uuid4()) + '.png'
    file_path = os.path.join(Img_Label_path, file_name)

    plt.savefig(file_path)
    plt.close()

    '''Label'''
    csv_file = os.path.join(Img_Label_path, 'label.csv')
    value = Label

    if os.path.exists(csv_file):
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([file_name, value])
    else:
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([file_name, value])
