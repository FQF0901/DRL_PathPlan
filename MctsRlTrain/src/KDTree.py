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
        self.grid_state = {}  # Dictionary to store grid states with their occupied value
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

    '''更新网格状态，occupied 为 0 到 1 的数值'''
    def add_or_update_grid(self, x, y, yaw_rad, occupied):
        x_idx, y_idx, yaw_idx = self._to_grid_indices(x, y, yaw_rad)
        grid_key = (x_idx, y_idx, yaw_idx)

        grid_pos = self._from_grid_indices(x_idx, y_idx, yaw_idx)

        if occupied > 0:  # If occupied is greater than 0, update or add the grid
            self.grid_state[grid_key] = occupied
            if not any(np.all(grid_pos == p) for p in self.occupied_grids):
                self.occupied_grids = np.vstack([self.occupied_grids, grid_pos])  # Add new grid
                self.kd_tree = KDTree(self.occupied_grids)  # Rebuild KDTree with updated occupied grids
        else:  # If occupied is 0, remove the grid if it exists
            if grid_key in self.grid_state:
                del self.grid_state[grid_key]
                self.occupied_grids = np.array([p for p in self.occupied_grids if not np.array_equal(p, grid_pos)])  # Remove grid
                self.kd_tree = KDTree(self.occupied_grids)  # Rebuild KDTree with updated occupied grids

    '''检查指定网格索引 (x_idx, y_idx, yaw_idx) 对应的网格的占用程度'''
    def is_occupied(self, x, y, yaw_rad):
        x_idx, y_idx, yaw_idx = self._to_grid_indices(x, y, yaw_rad)
        return self.grid_state.get((x_idx, y_idx, yaw_idx), 0.0)  # Default to 0.0 if not found

    '''根据给定的位置 (x, y, yaw) 查找附近的网格'''
    def locate_grid(self, x, y, yaw):
        x_idx, y_idx, yaw_idx = self._to_grid_indices(x, y, yaw)
        grid_pos = self._from_grid_indices(x_idx, y_idx, yaw_idx)
        return self.kd_tree.query_ball_point(grid_pos, r=0.5)  # Adjust radius if needed

# -------------------------- Test --------------------------
if __name__ == "__main__":

    kd_tree_map = KdTreeGridMap()

    x, y, yaw_rad = 1.0, 2.0, 0.5

    kd_tree_map.add_or_update_grid(x, y, yaw_rad, occupied=0.1)

    x, y, yaw_rad = 1.04, 2.04, 0.5
    is_occupied = kd_tree_map.is_occupied(x, y, yaw_rad)    # 0: Non-occupied, (0, 1]: occupied_value

    print("网格是否被占用:", is_occupied)
