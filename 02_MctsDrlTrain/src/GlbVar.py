"""
@author: Fqf
@time: 20240618
@file: GlbVar.py
@description: Global shared variables or types
"""
# ==========================================================
# ====================== Global Types ======================
# ==========================================================

# -------------------- Node Type --------------------
class Node:
    def __init__(self, x = 0, y = 0, yaw_rad = 0) -> None:
        self.x = x
        self.y = y
        self.yaw_rad = yaw_rad

# -------------------- HasNode Type --------------------
class HasNode:
    def __init__(self, x = 0, y = 0, yaw_rad = 0, g_cost = 0, h_cost = 0, parent = None, children = None) -> None:
        self.node = Node(x, y, yaw_rad)
        self.parent = parent
        self.children = children if children is not None else []
        self.g_cost = g_cost
        self.h_cost = h_cost

# -------------------- MctsNode Type --------------------
class MctsNode:
    def __init__(self, x = 0, y = 0, yaw_rad = 0, parent = None, children = None, TreeId = 0, TreeLvl = 0, \
                 DnnV = 0, DnnP = 0, n_visit = 0, Vdone = False, Type = 1, ActInfo = None, ActCost = 0) -> None:
        self.node = Node(x, y, yaw_rad)
        self.parent = parent
        self.children = children if children is not None else []
        self.TreeId = TreeId    # 0:Root, 1:Left tree, 2:Right tree
        self.TreeLvl = TreeLvl
        self.Value = DnnV
        self.Probs = DnnP
        self.n_visit = n_visit
        self.Vdone = Vdone
        self.type = Type    # 0:default, 1:Unexplored, 2:Ovlp, 3:RepeatMove, 4:PathFnd
        self.ActInfo = ActInfo if ActInfo is not None else [0, 0, 0]    # Steer Gear Distance: Represents the action from the parent node to this node
        self.ActCost = ActCost

# ----------------------- PcptGeo -----------------------
class PcptGeo:
    def __init__(self, StPt = [], TgtPt = [], Slot = [], Obst_list = []) -> None:
        self.StartPoint = StPt
        self.TargetPoint = TgtPt
        self.Slot = Slot
        self.Obstcle_list = Obst_list

# ----------------------- GridMap ------------------------
class GridMap:
    def __init__(self, StPt = [], TgtPt = [], Map = []) -> None:
        self.StartPoint = StPt
        self.TargetPoint = TgtPt
        self.Map = Map

# ==========================================================
# ==================== Global Variables ====================
# ==========================================================
PcptInfo = PcptGeo()    # GridMap() [important]