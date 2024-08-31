"""
@author: Fqf
@time: 20240618
@file: Mcts.py
@description: Decision Tree
"""

import os
import numpy as np
import graphviz
import time
import GlbVar
import CollisionCheck
import DrlUtil
import KDTree
import DrlCfg
import matplotlib.cm as cm
import matplotlib.colors as colors
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from Util import Config

# ==========================================================
# ======================= Mcts Tree ========================
# ==========================================================
class MctsTree:
    def __init__(self) -> None:
        self.exploration_weight = 0.5    # Keep the chance average at first (larger), then gradually move closer to the value (smaller)
        self.gamma = 0.99
        self.expd_maxcnt = 750

        self.RootMctsNode = GlbVar.MctsNode()
        self.TargetPose = GlbVar.Node()

        # self.GridMap = GlbVar.NodeGridMap()
        self.GridMap = KDTree.KdTreeGridMap()

# ----------------------- Select Node ----------------------
    """Select node(type should equal to 1) among the child by PUCT"""
    def SelectNode(self, crnt_node):
        best_value = float("-inf")
        selected_node = GlbVar.MctsNode()

        # 1. Determine whether the type of crnt_node is correct(actually determine the type of root_node)
        if crnt_node.type != 1: # 0:default, 1:Unexplored, 2:Ovlp, 3:RepeatMove, 4:PathFnd
            return crnt_node, crnt_node.type

        # 2. If the type of root_node is ok, the selection strategy is executed
        for child_node in crnt_node.children:
            exploitation_term = child_node.Value
            exploration_term = np.sqrt(np.log2(child_node.parent.n_visit) / (1 + child_node.n_visit))
            puct_value = exploitation_term + self.exploration_weight * exploration_term  # UCT

            if puct_value > best_value and child_node.type == 1:    # Node with type that Ovlp/RepeatMove/PathFnd should not be selected and expanded
                best_value = puct_value
                selected_node = child_node

        selected_node.n_visit = selected_node.n_visit + 1
        
        return selected_node, selected_node.type

# ----------------------- Expand Node ----------------------
    """Expand the node and init the related info"""
    def ExpandNode(self, crnt_node, policy_value_net):

        # 1. Expand Node by bicyc-model
        Expd_MctsNode_list = []
        new_node_list, ExpdAct_list = DrlUtil.Expd_Node(crnt_node.node)

        Child_Value = DrlUtil.NodeNetValue(crnt_node.node, policy_value_net)

        # 2. Add element info into expanded node
        for index, node in enumerate(new_node_list, start=1):
            NewMctsNode = GlbVar.MctsNode()

            # 2.1 New Mcts Node Element Info
            NewMctsNode.node = node
            NewMctsNode.parent = crnt_node
            NewMctsNode.TreeLvl = NewMctsNode.parent.TreeLvl + 1
            NewMctsNode.Value = Child_Value[0][index-1]
            NewMctsNode.Probs = 0

            # 2.2 New Mcts Node ActInfo
            StrAct = ExpdAct_list[index-1][0]
            GearAct = ExpdAct_list[index-1][1]
            DistAct = ExpdAct_list[index-1][2]
            NewMctsNode.ActInfo = [StrAct, GearAct, DistAct]

            # 2.3 New Mcts Node ActCost
            NewMctsNode.ActCost = - abs(crnt_node.ActInfo[0] - NewMctsNode.ActInfo[0]) * DrlCfg.TreePara.SteerSpdCostCoff \
                                - abs(crnt_node.ActInfo[1] - NewMctsNode.ActInfo[1]) * DrlCfg.TreePara.GearShiftCostCoff \
                                - DrlCfg.TreePara.ExpdNodeCost

            # 2.4 New Mcts Node Type
            if CollisionCheck.CollisionCheck_opt(node): # DrlUtil.IsOvlpAllObst(node)   # used for debug [important]
                NewMctsNode.type = 2    # 2:Ovlp
            elif DrlUtil.ChkRepeatAct(NewMctsNode):
                NewMctsNode.type = 3    # 3:RepeatMove (Impossible that collision and repeat both satisfied)
            elif self.GridMap.is_occupied(node.x, node.y, node.yaw_rad):
                NewMctsNode.type = 5    # 5:Occupied grid
            else:
                self.GridMap.add_or_update_grid(node.x, node.y, node.yaw_rad, True)
                NewMctsNode.type = 1    # 1:Unexplored

            # 3. return expand node list
            crnt_node.children.append(NewMctsNode)
            Expd_MctsNode_list.append(NewMctsNode)    

        return Expd_MctsNode_list

# ---------------------- Backpropagate ---------------------
    """Back propagation, update the node type"""
    def BackpropagateType(self, node):
        # 1. The end of the recursion: it has recursed to the root node
        if node.parent == None:
            return
        
        node_bro_PathFnd = False
        for node_bro in node.parent.children:   # Find all bro nodes of this node
            if node_bro.type == 1:  # If have unexplored bro nodes, backtpropagate is terminate
                return
        
        # 2. Determine parent type based on bro nodes types
            elif node_bro.type == 4:    # If there are no unexplored bro nodes, but exist PathFnd bro nodes, the parent node should be pathfnd
                node_bro_PathFnd = True

        node.parent.type = 4 if node_bro_PathFnd else 2

        # 3. backtpropagation
        self.BackpropagateType(node.parent)
    
    """Back propagation, update the node value after episode end (its input is root node)"""
    def BackpropagateValue(self, node):
        # 1. The end of the recursion
        if node.n_visit == 0:   # True value leaf node
            node.Vprev = node.Value
            node.Value = DrlUtil.LeafNodeValueJudge(node)
            node.Vdone = True
            return
        elif node.n_visit == 1:   # True value leaf node
            node.Vprev = node.Value
            node.Value = DrlUtil.LeafNodeValueJudge(node)
            node.Vdone = True

        # 2. backpropagation: if the Vdone of the child node is False, update the Value of the child node first
        elif node.n_visit > 1:
            for child_node in node.children:
                if not child_node.Vdone:
                    self.BackpropagateValue(child_node)

        # 3. If the Vdone of all child nodes is True, calculate the value of this node
        if not(node.n_visit == 1 and node.type == 4):
            all_child_Vdone = all(child_node.Vdone for child_node in node.children)

            if all_child_Vdone:
                # 3.1 Cal num of valid child
                node_valid_child_num = 0
                for child_node in node.children:
                    if not child_node.type == 3:
                        node_valid_child_num = node_valid_child_num + 1

                # 3.2 Cal node.value
                node_value = 0
                for child_node in node.children:
                    itmdt_reward = 0
                    child_node.Probs = child_node.n_visit / (node.n_visit - 1) # 1 / node_valid_child_num # (DrlCfg.TreePara.DistActDim * DrlCfg.TreePara.GearActDim * DrlCfg.TreePara.StrActDim) # child_node.n_visit / (node.n_visit - 1)
                    node_value = node_value + (itmdt_reward + self.gamma * child_node.Value) * child_node.Probs

                node.Vprev = node.Value
                node.Value = node_value
                node.Vdone = True
    
# ----------------------- Simulation -----------------------
    """This is a complete exploration process for root node"""
    def Simulate(self, expd_maxcnt, policy_value_net):

        for cnt in range(expd_maxcnt):
            self.expd_maxcnt = expd_maxcnt
            node = self.RootMctsNode
            node.n_visit = node.n_visit + 1

            # 1. Select Node
            # 1.1 Explore and select until a leaf node is found
            while True:
                if DrlUtil.IsLeafNode(node):
                    SelectedLeafNode = node
                    break
                    
                selected_node, SelectNodeInfo = self.SelectNode(node)    # select node whose type = 1

                # 1.2 Execute different strategies according to different node types, used to avoid select dead nodes in the next round of simulation: 2:Ovlp, 3:RepeatMove, 4:PathFnd
                if SelectNodeInfo == 2: # Case for root node
                    self.RootMctsNode.n_visit = self.RootMctsNode.n_visit - 1
                    _ = self.BackpropagateValue(self.RootMctsNode)
                    # print('-- Episode end! All children of the root node overlap, so openlist = [] and cannot expand further')
                    return SelectNodeInfo, cnt
                elif SelectNodeInfo == 4:   # Case for root node
                    self.RootMctsNode.n_visit = self.RootMctsNode.n_visit - 1
                    _ = self.BackpropagateValue(self.RootMctsNode)
                    # print('-- Episode end! All children of the root node have been explored, and some whose type are PathFnd')
                    return SelectNodeInfo, cnt

                node = selected_node

            PathFnd, _ = DrlUtil.CalValidRS(SelectedLeafNode.node, self.TargetPose)
            if PathFnd:
                SelectedLeafNode.type = 4

            _ = self.BackpropagateType(SelectedLeafNode)

            # 2. Expand Node based on leaf node.
            Expd_MctsNode_list = self.ExpandNode(SelectedLeafNode, policy_value_net)

            # 3. Backprogagete node info such as type and value
            _ = self.BackpropagateType(Expd_MctsNode_list[0])

        _ = self.BackpropagateValue(self.RootMctsNode)
        # print('-- Episode end due to reach self.expd_maxcnt !')
        return SelectNodeInfo, cnt

# ---------------------- StoreTreeInfo ---------------------
    """Store the tree information after backpropagate value"""
    # External packaging interface
    def StoreTreeInfo(self):
        state_list, value_list = [], []
        GlbVar.vis_node_list.clear()
        self.TravslTreeInfo(self.RootMctsNode, state_list, value_list)

        return state_list, value_list

    # Recursively traverse the entire tree
    def TravslTreeInfo(self, MctsNode, state_list, value_list):
        stack = [MctsNode]
    
        while stack:
            node = stack.pop()
            
            for child_node in node.children:
                # Only node with full exploration or high value(0.4) should be recorded and learned
                if (((child_node.n_visit >= max(6, self.expd_maxcnt / 125)) or (child_node.n_visit >= 1 and child_node.Value > 0.2)) 
                    and (not child_node.Store)):
                    
                    state_list.append([child_node.node, self.TargetPose, GlbVar.PcptInfo])
                    value_list.append([GrandChild.Value for GrandChild in child_node.children])

                    GlbVar.vis_node_list.add_node(child_node.node, child_node.Value)

                    child_node.Store = True
                    stack.append(child_node)

# ---------------------- Visualization ---------------------
    def VisTree(self, scene_pkl_file='', time_slice_idx=''):
        root = self.RootMctsNode
        store_path = Config.StorePath.tree_info_path

        if not scene_pkl_file=='' and not time_slice_idx=='':
            filename = f"{os.path.basename(scene_pkl_file).rsplit('.', 1)[0]}_rowidx{time_slice_idx}"
        else:
            filename = 'VisTree'

        dot = graphviz.Digraph()
        dot.attr(rankdir='LR')
        self.add_nodes(root, dot)
        full_path = os.path.join(store_path, filename)
        dot.render(full_path, format='svg', cleanup=True)    

    def add_nodes(self, node, dot):
        formatted_x = "{:.3f}".format(node.node.x)
        formatted_y = "{:.3f}".format(node.node.y)
        formatted_yaw_rad = "{:.3f}".format(node.node.yaw_rad)
        formatted_V = "{:.2f}".format(node.Value)
        formatted_Vprev = "{:.2f}".format(node.Vprev)
        formatted_Probs = "{:.2f}".format(node.Probs)
        formatted_ActInfo = ", ".join("{:.2f}".format(val) for val in node.ActInfo)
        formatted_ActCost = "{:.2f}".format(node.ActCost)
        
        label = f"({formatted_x}, {formatted_y}, {formatted_yaw_rad})\
            \nValue: {formatted_V}, Probs: {formatted_Probs}, n_visit: {node.n_visit}, Type: {node.type}, Vdone: {node.Vdone}, Store: {node.Store}\
            \nVprev: {formatted_Vprev}, TreeLvl: {node.TreeLvl}, ActInfo: {formatted_ActInfo}, ActCost: {formatted_ActCost}"
        
        if node.type == 2:  # ovlp node
            fillcolor = "Magenta"
        elif node.type == 3:    # RepeatMove
            fillcolor = "yellow"
        elif node.type == 4:    # PathFnd node
            fillcolor = "lightgreen"
        elif node.type == 5:    # Occupied grid
            fillcolor = "gray"
        elif node.n_visit == 0: # un-visit node
            fillcolor = "white"
        else:
            fillcolor = self.get_lightblue_color(node.Value)    # Select but non-PathFnd node

        dot.node(str(id(node)), label, shape="box", style="filled", fillcolor=fillcolor)

        for child in node.children:
            dot.edge(str(id(node)), str(id(child)))
            self.add_nodes(child, dot)

    def get_lightblue_color(self, value):
        # Define a colormap (here using 'Blues') and a normalization for values between 0 and 1
        # The node.Value range is 0~120, but the visualization range should be larger to prevent the color from being too dark or too light.
        cmap = cm.Blues
        norm = colors.Normalize(vmin=-0.2, vmax=1.2)
        
        rgba_color = cmap(norm(value))  # Map value to a color in the colormap
        hex_color = colors.rgb2hex(rgba_color)  # Convert RGBA tuple to hexadecimal color code

        return hex_color


# -------------------------- Test --------------------------
# import Mcts

# MT = Mcts.MctsTree()
# MT.Simulate()
# MT.VisTree(MT.RootMctsNode)