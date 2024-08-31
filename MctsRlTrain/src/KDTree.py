"""
@author: Fqf
@time: 20240828
@file: KDTree.py
@description: Accelerate occupation grid map localization
"""

import numpy as np
from scipy.spatial import KDTree
import DrlCfg


# ==========================================================
# ====================== KdTree Search =====================
# ==========================================================

class KdTreeGridMap:
    def __init__(self):
        params = DrlCfg.NodeGridMapParams()
        self.x_min = params.x_m_min
        self.x_max = params.x_m_max
        self.y_min = params.y_m_min
        self.y_max = params.y_m_max
        self.yaw_min = params.yaw_rad_min
        self.yaw_max = params.yaw_rad_max
        self.grid_size_x = params.grid_size_x_m
        self.grid_size_y = params.grid_size_y_m
        self.grid_size_yaw = params.grid_size_yaw_rad
        
        self.occupied_grids = np.empty((0, 3))  # Initialize as an empty array with 3 columns
        self.grid_state = {}
        self.kd_tree = KDTree(self.occupied_grids)  # Initialize KDTree with empty data

    '''将实际坐标 (x, y, yaw) 转换为网格索引'''
    def _to_grid_indices(self, x, y, yaw):
        x_idx = int((x - self.x_min) / self.grid_size_x)
        y_idx = int((y - self.y_min) / self.grid_size_y)
        yaw_idx = int((yaw - self.yaw_min) / self.grid_size_yaw)
        return x_idx, y_idx, yaw_idx

    '''将网格索引转换回实际坐标 (x, y, yaw)'''
    def _from_grid_indices(self, x_idx, y_idx, yaw_idx):
        x = self.x_min + x_idx * self.grid_size_x
        y = self.y_min + y_idx * self.grid_size_y
        yaw = self.yaw_min + yaw_idx * self.grid_size_yaw
        return x, y, yaw

    '''如果 occupied 为 True，则将网格添加到 KDTree 中；如果 False，则从 KDTree 中删除该网格'''
    def add_or_update_grid(self, x, y, yaw_rad, occupied):
        x_idx, y_idx, yaw_idx = self._to_grid_indices(x, y, yaw_rad)

        grid_key = (x_idx, y_idx, yaw_idx)
        if occupied:
            if grid_key not in self.grid_state or not self.grid_state[grid_key]:
                self.grid_state[grid_key] = True
                grid_pos = self._from_grid_indices(x_idx, y_idx, yaw_idx)
                self.occupied_grids = np.vstack([self.occupied_grids, grid_pos])  # Add new grid
                self.kd_tree = KDTree(self.occupied_grids)  # Rebuild KDTree with updated occupied grids
        else:
            if grid_key in self.grid_state and self.grid_state[grid_key]:
                self.grid_state[grid_key] = False
                grid_pos = self._from_grid_indices(x_idx, y_idx, yaw_idx)
                self.occupied_grids = np.array([p for p in self.occupied_grids if not np.array_equal(p, grid_pos)])  # Remove grid
                self.kd_tree = KDTree(self.occupied_grids)  # Rebuild KDTree with updated occupied grids

    '''检查指定网格索引 (x_idx, y_idx, yaw_idx) 对应的网格是否被占用'''
    def is_occupied(self, x, y, yaw_rad):
        x_idx, y_idx, yaw_idx = self._to_grid_indices(x, y, yaw_rad)

        return self.grid_state.get((x_idx, y_idx, yaw_idx), False)

    '''根据给定的位置 (x, y, yaw) 查找附近的网格'''
    def locate_grid(self, x, y, yaw):
        x_idx, y_idx, yaw_idx = self._to_grid_indices(x, y, yaw)
        grid_pos = self._from_grid_indices(x_idx, y_idx, yaw_idx)
        return self.kd_tree.query_ball_point(grid_pos, r=0.5)  # Adjust radius if needed

# -------------------------- Test --------------------------
if __name__ == "__main__":

    kd_tree_map = KdTreeGridMap()

    x, y, yaw_rad = 1.0, 2.0, 0.5

    kd_tree_map.add_or_update_grid(x, y, yaw_rad, occupied=True)

    x, y, yaw_rad = 1.04, 2.04, 0.5
    is_occupied = kd_tree_map.is_occupied(x, y, yaw_rad)

    print("网格是否被占用:", is_occupied)
