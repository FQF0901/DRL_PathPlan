"""
@author: Fqf
@time: 20240618
@file: GlbVar.py
@description: Global shared variables or types
"""

import scipy.sparse as sp
import DrlCfg

# ==========================================================
# ====================== Global Types ======================
# ==========================================================

# ----------------------- Node Type ------------------------
class Node:
    def __init__(self, x = 0, y = 0, yaw_rad = 0) -> None:
        self.x = x
        self.y = y
        self.yaw_rad = yaw_rad

# ---------------------- HasNode Type ----------------------
class HasNode:
    def __init__(self, x = 0, y = 0, yaw_rad = 0, g_cost = 0, h_cost = 0, parent = None, children = None) -> None:
        self.node = Node(x, y, yaw_rad)
        self.parent = parent
        self.children = children if children is not None else []
        self.g_cost = g_cost
        self.h_cost = h_cost

# ---------------------- MctsNode Type ---------------------
class MctsNode:
    def __init__(self, x = 0, y = 0, yaw_rad = 0, parent = None, children = None, TreeLvl = 0, Type = 1, \
                 DnnV = 0, Probs = 0, n_visit = 0, Vdone = False, Vprev = 0, Store = False, ActInfo = None, ActCost = 0) -> None:
        self.node = Node(x, y, yaw_rad)
        self.parent = parent
        self.children = children if children is not None else []
        self.TreeLvl = TreeLvl
        self.type = Type    # 0:default, 1:Unexplored, 2:Ovlp, 3:RepeatMove, 4:PathFnd, 5:Occupied grid
        self.Value = DnnV
        self.Probs = Probs  # DNN no longer provides policy
        self.n_visit = n_visit
        self.Vdone = Vdone
        self.Vprev = Vprev
        self.Store = Store
        self.ActInfo = ActInfo if ActInfo is not None else [0, 0, 0]    # Steer Gear Distance: Represents the action from the parent node to this node
        self.ActCost = ActCost

# -------------------- MctsEfctNode Type -------------------
class MctsEfctNode:
    def __init__(self, x = 0, y = 0, yaw_rad = 0, parent = None, children = None, DnnV = 0, Type = 1, \
                 n_visit = 0, Vdone = False, Store = False, TreeLvl = 0, ActInfo = None, ActCost = 0) -> None:
        self.node = Node(x, y, yaw_rad)
        self.parent = parent
        self.children = children if children is not None else []
        self.Value = DnnV
        self.type = Type    # 0:default, 1:Unexplored, 2:Ovlp, 3:RepeatMove, 4:PathFnd, 5:Occupied grid
        self.n_visit = n_visit
        self.Vdone = Vdone
        self.Store = Store
        self.TreeLvl = TreeLvl
        self.ActInfo = ActInfo if ActInfo is not None else [0, 0, 0]    # Steer Gear Distance: Represents the action from the parent node to this node
        self.ActCost = ActCost

class MctsBiDirNode:
    def __init__(self, x = 0, y = 0, yaw_rad = 0, parent = None, children = None, TreeId = 0, TreeLvl = 0, \
                 DnnV = 0, Probs = 0, n_visit = 0, Vdone = False, Type = 1, ActInfo = None, ActCost = 0) -> None:
        self.node = Node(x, y, yaw_rad)
        self.parent = parent
        self.children = children if children is not None else []
        self.TreeId = TreeId    # 0:Root, 1:Left tree, 2:Right tree
        self.TreeLvl = TreeLvl
        self.Value = DnnV
        self.Probs = Probs  # DNN no longer provides policy
        self.n_visit = n_visit
        self.Vdone = Vdone
        self.type = Type    # 0:default, 1:Unexplored, 2:Ovlp, 3:RepeatMove, 4:PathFnd, 5:Occupied grid
        self.ActInfo = ActInfo if ActInfo is not None else [0, 0, 0]    # Steer Gear Distance: Represents the action from the parent node to this node
        self.ActCost = ActCost

# ------------------ Node Expand Grid Map ------------------
class NodeGridMap:
    def __init__(self) -> None:
        self.num_grids_x = int((DrlCfg.NodeGridMapParam.x_m_max - DrlCfg.NodeGridMapParam.x_m_min) / DrlCfg.NodeGridMapParam.grid_size_x_m)
        self.num_grids_y = int((DrlCfg.NodeGridMapParam.y_m_max - DrlCfg.NodeGridMapParam.y_m_min) / DrlCfg.NodeGridMapParam.grid_size_y_m)
        self.num_grids_yaw_rad = int((DrlCfg.NodeGridMapParam.yaw_rad_max - DrlCfg.NodeGridMapParam.yaw_rad_min) / DrlCfg.NodeGridMapParam.grid_size_yaw_rad)

        self.data = []
        self.row_indices = []
        self.col_indices = []

        # Sparse Matrix
        self.grid_type_sparse = sp.coo_matrix((self.data, (self.row_indices, self.col_indices)),
                                shape=(self.num_grids_x * self.num_grids_y * self.num_grids_yaw_rad, self.num_grids_x * self.num_grids_y * self.num_grids_yaw_rad))

    def reset_matrix(self):
        self.data = []
        self.row_indices = []
        self.col_indices = []
        self.grid_type_sparse = sp.coo_matrix((self.data, (self.row_indices, self.col_indices)),
                                shape=(self.num_grids_x * self.num_grids_y * self.num_grids_yaw_rad, self.num_grids_x * self.num_grids_y * self.num_grids_yaw_rad))

    def get_grid_indices(self, x, y, yaw):
        if not (DrlCfg.NodeGridMapParam.x_m_min <= x <= DrlCfg.NodeGridMapParam.x_m_max 
                and DrlCfg.NodeGridMapParam.y_m_min <= y <= DrlCfg.NodeGridMapParam.y_m_max 
                and DrlCfg.NodeGridMapParam.yaw_rad_min <= yaw <= DrlCfg.NodeGridMapParam.yaw_rad_max):
            raise ValueError("坐标超出网格范围")
        
        ix = int((x - DrlCfg.NodeGridMapParam.x_m_min) / DrlCfg.NodeGridMapParam.grid_size_x_m)
        iy = int((y - DrlCfg.NodeGridMapParam.y_m_min) / DrlCfg.NodeGridMapParam.grid_size_y_m)
        iyaw = int((yaw - DrlCfg.NodeGridMapParam.yaw_rad_min) / DrlCfg.NodeGridMapParam.grid_size_yaw_rad)
        return ix, iy, iyaw

    def mark_grid_as_occupied(self, x, y, yaw):
        ix, iy, iyaw = self.get_grid_indices(x, y, yaw)
        idx = ix * self.num_grids_y * self.num_grids_yaw_rad + iy * self.num_grids_yaw_rad + iyaw
        self.data.append(1)
        self.row_indices.append(idx)
        self.col_indices.append(idx)
        self.grid_type_sparse = sp.coo_matrix((self.data, (self.row_indices, self.col_indices)),
                                    shape=(self.num_grids_x * self.num_grids_y * self.num_grids_yaw_rad, self.num_grids_x * self.num_grids_y * self.num_grids_yaw_rad))

    def check_grid_status(self, x, y, yaw):
        ix, iy, iyaw = self.get_grid_indices(x, y, yaw)
        idx = ix * self.num_grids_y * self.num_grids_yaw_rad + iy * self.num_grids_yaw_rad + iyaw
        grid_type_sparse_csc = self.grid_type_sparse.tocsc()

        return grid_type_sparse_csc[idx, idx] > 0


# ---------------------- VisNode Type ----------------------
class VisNode:
    def __init__(self, node, value):
        self.node = node
        self.value = value

class VisNodeList:
    def __init__(self):
        self.node_list = []

    def add_node(self, node, value):
        vis_node = VisNode(node, value)
        self.node_list.append(vis_node)

    def get_data(self):
        x = [vis_node.node.x for vis_node in self.node_list]
        y = [vis_node.node.y for vis_node in self.node_list]
        values = [vis_node.value for vis_node in self.node_list]
        return x, y, values
    
    def clear(self):
        self.node_list = []

# ------------------------ PcptGeo -------------------------
class PcptGeo:
    def __init__(self, StPt = [], TgtPt = [], Slot = [], Obst_list = []) -> None:
        self.StartPoint = StPt
        self.TargetPoint = TgtPt
        self.Slot = Slot
        self.Obstcle_list = Obst_list

    def clear(self):
        self.StartPoint = []
        self.TargetPoint = []
        self.Slot = []
        self.Obstcle_list = []

# ------------------------ GridMap -------------------------
class GridMap:
    def __init__(self, StPt = [], TgtPt = [], Map = []) -> None:
        self.StartPoint = StPt
        self.TargetPoint = TgtPt
        self.Map = Map


# ==========================================================
# ==================== Global Variables ====================
# ==========================================================
PcptInfo = PcptGeo()    # GridMap() [important]
vis_node_list = VisNodeList()
