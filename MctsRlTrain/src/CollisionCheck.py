"""
@author: Fqf
@time: 20240830
@file: CollisionCheck.py
@description: Supports overlap checking for compilation acceleration
"""
from numba import jit, njit
import numpy as np
import GlbVar
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import Config


# ==========================================================
# ===================== CollisionCheck =====================
# ==========================================================

@njit
def GetVehRect_opt(crnt_node_x, crnt_node_y, crnt_node_yaw_rad, LatMargin, LgtMargin, 
                   VehicleLength, VehicleWidth, Center2RearAxle):
    Rear2BackDist = VehicleLength / 2 - Center2RearAxle
    Rear2FrtDistDist = VehicleLength / 2 + Center2RearAxle

    local_corners = np.array([
        [-Rear2BackDist - LgtMargin, -VehicleWidth / 2 - LatMargin],
        [-Rear2BackDist - LgtMargin, VehicleWidth / 2 + LatMargin],
        [Rear2FrtDistDist + LgtMargin, VehicleWidth / 2 + LatMargin],
        [Rear2FrtDistDist + LgtMargin, -VehicleWidth / 2 - LatMargin]
    ], dtype=np.float32)

    cos_yaw = np.cos(crnt_node_yaw_rad)
    sin_yaw = np.sin(crnt_node_yaw_rad)
    
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ], dtype=np.float32)
    
    rotated_corners = np.empty_like(local_corners)
    for i in range(local_corners.shape[0]):
        rotated_corners[i, 0] = local_corners[i, 0] * rotation_matrix[0, 0] + local_corners[i, 1] * rotation_matrix[0, 1]
        rotated_corners[i, 1] = local_corners[i, 0] * rotation_matrix[1, 0] + local_corners[i, 1] * rotation_matrix[1, 1]
    
    global_corners = rotated_corners + np.array([crnt_node_x, crnt_node_y], dtype=np.float32)

    return global_corners

@njit
def GetProjection_opt(corners, axis):
    min_proj = np.inf
    max_proj = -np.inf
    
    for i in range(corners.shape[0]):
        proj = corners[i, 0] * axis[0] + corners[i, 1] * axis[1]
        if proj < min_proj:
            min_proj = proj
        if proj > max_proj:
            max_proj = proj
    
    return min_proj, max_proj

@njit
def ChkOvlp_opt(proj1, proj2):
    return not (proj1[1] < proj2[0] or proj2[1] < proj1[0])

@njit
def IsOvlpSprtAxis_opt(rect1, rect2):
    num_edges = len(rect1)
    axes = np.empty((2 * num_edges, 2), dtype=np.float32)

    for i in range(num_edges):
        edge = rect1[i] - rect1[(i-1) % num_edges]
        normal = np.array([-edge[1], edge[0]], dtype=np.float32)
        norm = np.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-10:
            axes[i] = normal / norm

    num_rect2_edges = len(rect2)
    for i in range(num_rect2_edges):
        edge = rect2[i] - rect2[(i-1) % num_rect2_edges]
        normal = np.array([-edge[1], edge[0]], dtype=np.float32)
        norm = np.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-10:
            axes[num_edges + i] = normal / norm
    
    for axis in axes:
        proj1 = GetProjection_opt(rect1, axis)
        proj2 = GetProjection_opt(rect2, axis)
        if not ChkOvlp_opt(proj1, proj2):
            return False
    return True

@njit
def IsOvlpAllObst_opt(crnt_node_x, crnt_node_y, crnt_node_yaw_rad, LatMargin, LgtMargin, ObstRect_array,
                      VehicleLength, Center2RearAxle, VehicleWidth):
    VehRect = GetVehRect_opt(crnt_node_x, crnt_node_y, crnt_node_yaw_rad, LatMargin, LgtMargin,
                              VehicleLength, VehicleWidth, Center2RearAxle)

    num_obstacles = ObstRect_array.shape[0]
    
    for i in range(num_obstacles):
        ObstRect = ObstRect_array[i]
        if IsOvlpSprtAxis_opt(ObstRect, VehRect):
            return True
    return False

def CollisionCheck_opt(crnt_node, LatMargin = Config.VehPara.LatMargin, LgtMargin = Config.VehPara.LgtMargin):
    crnt_node_x = crnt_node.x
    crnt_node_y = crnt_node.y
    crnt_node_yaw_rad = crnt_node.yaw_rad

    ObstRect_array = np.array(GlbVar.PcptInfo.Obstcle_list, dtype=np.float32)

    return IsOvlpAllObst_opt(crnt_node_x, crnt_node_y, crnt_node_yaw_rad, LatMargin, LgtMargin, ObstRect_array, 
                              Config.VehPara.VehicleLength, Config.VehPara.Center2RearAxle, Config.VehPara.VehicleWidth)
