"""
@author: Fqf
@time: 20240618
@file: DrlUtil.py
@description: Shared Libraries for DRL
"""

import math
import Config
import GlbVar
import RS
import numpy as np

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
    Rear2BackDist = Config.VehPara.VehicleLength / 2 + (Config.VehPara.Center2RearAxle)
    Rear2FrtDistDist = Config.VehPara.VehicleLength / 2 - (Config.VehPara.Center2RearAxle)

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
            return False, [], []
        
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
    
    for gear in Config.TreePara.GearAct_list:
        for StrAngRate in Config.TreePara.StrAct_list:
            for dist in Config.TreePara.ExpdStepLen_list:
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
    TreeLevelCost = node.TreeLvl * Config.TreePara.TreeLvlCostCoff
    ActCost = node.ActCost
    Close2ObstcCost = 0 # need to update [important]
    PathFndReward = 100 if node.type == 4 else 0

    TotalValue = TreeLevelCost + ActCost + Close2ObstcCost + PathFndReward

    return TotalValue
