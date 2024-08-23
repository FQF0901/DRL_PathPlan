"""
@author: Fqf
@time: 20240618
@file: MctsBiDir.py
@description: Decision Tree [This document is temporarily invalid]
"""

import os
import numpy as np
import graphviz
import GlbVar
import DrlUtil
import random
import DrlCfg

# ==========================================================
# ===================== MctsBiDir Tree =====================
# ==========================================================
class MctsBiDirTree:
    def __init__(self) -> None:
        self.exploration_weight = 10
        self.gamma = 0.99
        self.expd_maxcnt = 3000

        # Bi-directional expansion tree
        self.RtSpMctsBiDirNode = GlbVar.MctsNode() # 0:Root, 1:Left tree, 2:Right tree
        self.RtSpMctsBiDirNode.TreeId = 1
        self.RtSpMctsBiDirNode.TreeLvl = 1

        self.RtTpMctsBiDirNode = GlbVar.MctsNode()
        self.RtTpMctsBiDirNode.TreeId = 2
        self.RtTpMctsBiDirNode.TreeLvl = 1

        self.RootMctsBiDirNode = GlbVar.MctsNode()
        self.RootMctsBiDirNode.children.append(self.RtSpMctsBiDirNode)
        self.RootMctsBiDirNode.children.append(self.RtTpMctsBiDirNode)

        self.RtSpMctsBiDirNode.parent = self.RootMctsBiDirNode
        self.RtTpMctsBiDirNode.parent = self.RootMctsBiDirNode

        self.SpTreeSelectNode = self.RtSpMctsBiDirNode   # RS should connect SpTreeSelectNode and TpTreeSelectNode
        self.TpTreeSelectNode = self.RtTpMctsBiDirNode

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
            exploitation_term = child_node.Value    # n_win / n_visit 或 DnnQ [important]
            exploration_term = np.sqrt(np.log2(child_node.parent.n_visit) / (1 + child_node.n_visit))
            puct_value = exploitation_term + self.exploration_weight * exploration_term  # UCT

            if puct_value > best_value and child_node.type == 1:   # Node with type that Ovlp/RepeatMove/PathFnd should not be selected and expanded
                best_value = puct_value
                selected_node = child_node

        selected_node.n_visit = selected_node.n_visit + 1
        
        return selected_node, selected_node.type

# ----------------------- Expand Node ----------------------
    """Expand the node and init the related info"""
    def ExpandNode(self, crnt_node):
        # 1. Expand Node by bicyc-model
        Expd_MctsBiDirNode_list = []
        new_node_list, ExpdAct_list = DrlUtil.Expd_Node(crnt_node.node)

        # 2. Add element info into expanded node
        cnt = 0
        for node in new_node_list:
            NewMctsBiDirNode = GlbVar.MctsNode()

            # 2.1 New MctsBiDir Node Element Info
            NewMctsBiDirNode.node = node
            NewMctsBiDirNode.parent = crnt_node
            NewMctsBiDirNode.TreeId = NewMctsBiDirNode.parent.TreeId
            NewMctsBiDirNode.TreeLvl = NewMctsBiDirNode.parent.TreeLvl + 1
            DnnV = random.uniform(0, 100)    # Need delete [important]
            NewMctsBiDirNode.Value = DnnV    # Need update [important]
            NewMctsBiDirNode.Probs = 0

            # 2.2 New MctsBiDir Node ActInfo
            StrAct = ExpdAct_list[cnt][0]
            GearAct = ExpdAct_list[cnt][1]
            DistAct = ExpdAct_list[cnt][2]
            NewMctsBiDirNode.ActInfo = [StrAct, GearAct, DistAct]
            cnt = cnt + 1

            # 2.3 New MctsBiDir Node ActCost
            NewMctsBiDirNode.ActCost = - abs(crnt_node.ActInfo[0] - NewMctsBiDirNode.ActInfo[0]) * DrlCfg.TreePara.SteerSpdCostCoff \
                                - abs(crnt_node.ActInfo[1] - NewMctsBiDirNode.ActInfo[1]) * DrlCfg.TreePara.GearShiftCostCoff

            # 2.4 New MctsBiDir Node Type
            if DrlUtil.IsOvlpAllObst(node):
                NewMctsBiDirNode.type = 2    # 2:Ovlp
            elif DrlUtil.ChkRepeatAct(NewMctsBiDirNode):
                NewMctsBiDirNode.type = 3    # 3:RepeatMove (Impossible that collision and repeat both satisfied)
            else:
                NewMctsBiDirNode.type = 1    # 1:Unexplored

            # 3. return expand node list
            crnt_node.children.append(NewMctsBiDirNode)
            Expd_MctsBiDirNode_list.append(NewMctsBiDirNode)
        
        return Expd_MctsBiDirNode_list

# ---------------------- Backpropagate ---------------------
    """Back propagation, update the node type"""
    def BackpropagateType(self, node):
        # 1. The end of the recursion: it has recursed to the root node
        if node.parent == None:
            return
        
        # 2. Determine parent type based on bro nodes types
        node_bro_list = node.parent.children    # Find all bro nodes of this node
        node_bro_PathFnd = False

        for node_bro in node_bro_list:
            if node_bro.type == 1:  # If have unexplored bro nodes, backtpropagate is terminate
                return
            elif node_bro.type == 4:    # If there are no unexplored bro nodes, but exist PathFnd bro nodes, the parent node should be pathfnd
                node_bro_PathFnd = True

        node.parent.type = 4 if node_bro_PathFnd else 2

        # 3. backtpropagation
        self.BackpropagateType(node.parent)
    
    """Back propagation, update the node value after episode end (its input is root node)"""
    def BackpropagateValue(self, node):
        # 1. The end of the recursion
        if node.n_visit == 1:   # True value leaf node
            node.Value = DrlUtil.LeafNodeValueJudge(node)   # need update [important]
            node.Vdone = True
            return
        elif node.n_visit == 0: # Not selected leaf node
            node.Value = 0 if node.type == 1 else 0    # Non-true value nodes should not be learned
            node.Vdone = True
            return
        
        # 2. backtpropagation: if the Vdone of the child node is False, update the Value of the child node first
        for child_node in node.children:
            if not child_node.Vdone:
                self.BackpropagateValue(child_node)

        # 3. If the Vdone of all child nodes is True, calculate the value of this node
        all_child_Vdone = all(child_node.Vdone for child_node in node.children)

        if all_child_Vdone:
            node_value = 0
            for child_node in node.children:
                itmdt_reward = 0
                child_node.Probs = child_node.n_visit / (node.n_visit - 1)
                node_value = node_value + (itmdt_reward + self.gamma * child_node.Value) * child_node.Probs


            node.Value = node_value
            node.Vdone = True
    
# ----------------------- Simulation -----------------------
    """This is a complete exploration process for root node"""
    def Simulate(self, expd_maxcnt=3000):
        self.expd_maxcnt = expd_maxcnt

        for cnt in range(self.expd_maxcnt):
            node = self.RootMctsBiDirNode
            node.n_visit = node.n_visit + 1

            # 1. Select Node
            # 1.1 Explore and select until a leaf node is found, and store the selected leaf node into RtSpMctsBiDirNode or RtTpMctsBiDirNode
            while True:
                if DrlUtil.IsLeafNode(node):
                    SelectedLeafNode = node
                    
                    if SelectedLeafNode.TreeId == 1:
                        self.SpTreeSelectNode = SelectedLeafNode
                    elif SelectedLeafNode.TreeId == 2:
                        self.TpTreeSelectNode = SelectedLeafNode
                    break
                    
                selected_node, SelectNodeInfo = self.SelectNode(node)    # select node whose type = 1

                # 1.2 Execute different strategies according to different node types, used to avoid select dead nodes in the next round of simulation: 2:Ovlp, 3:RepeatMove, 4:PathFnd
                if SelectNodeInfo == 2: # Case for root node
                    _ = self.BackpropagateValue(self.RootMctsBiDirNode)
                    print('-- Episode end! All children of the root node overlap, so openlist = [] and cannot expand further')
                    return SelectNodeInfo, cnt
                elif SelectNodeInfo == 4:   # Case for root node
                    _ = self.BackpropagateValue(self.RootMctsBiDirNode)
                    print('-- Episode end! All children of the root node have been explored, and some whose type are PathFnd')
                    return SelectNodeInfo, cnt

                node = selected_node

            if SelectNodeInfo == 1:   # Case for normal leaf node
                PathFnd, _ = DrlUtil.CalValidRS(self.SpTreeSelectNode.node, self.TpTreeSelectNode.node)
                if PathFnd:
                    self.SpTreeSelectNode.type = 4
                    self.TpTreeSelectNode.type = 4

                _ = self.BackpropagateType(self.SpTreeSelectNode)
                _ = self.BackpropagateType(self.TpTreeSelectNode)

            # 2. Expand Node based on leaf node
            Expd_MctsBiDirNode_list = self.ExpandNode(SelectedLeafNode)

            # 3. Backprogagete node info such as type and value
            _ = self.BackpropagateType(Expd_MctsBiDirNode_list[0])

        _ = self.BackpropagateValue(self.RootMctsBiDirNode)
        print('-- Episode end due to reach self.expd_maxcnt !')
        return SelectNodeInfo, cnt

# ---------------------- StoreTreeInfo ---------------------
    """Store the tree information after backpropagate value"""
    # External packaging interface
    def StoreTreeInfo(self):
        state_list, act_probs_list, V_value_list = [], [], []
        self.TravslTreeInfo(self.RootMctsBiDirNode, state_list, act_probs_list, V_value_list)

        return state_list, act_probs_list, V_value_list

    # Recursively traverse the entire tree
    def TravslTreeInfo(self, node, state_list, act_probs_list, V_value_list):
        if node.n_visit == 1:   # End point of recursion
            state_list.append(1)    # need update [important]
            act_probs_list.append(1)
            V_value_list.append(1) # state value, not action value

        for child in node.children:
            self.TravslTreeInfo(child, state_list, act_probs_list, V_value_list)

# ---------------------- Visualization ---------------------
    def VisTree(self, scene_pkl_file='', time_slice_idx='', store_path=os.getcwd()):
        root = self.RootMctsBiDirNode

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
        formatted_ActInfo = ", ".join("{:.2f}".format(val) for val in node.ActInfo)
        formatted_Probs = "{:.2f}".format(node.Probs)  # Add this line to format Probs
        formatted_ActCost = "{:.2f}".format(node.ActCost)
        
        label = f"({formatted_x}, {formatted_y}, {formatted_yaw_rad})\
            \nType: {node.type}, Value: {formatted_V}, Probs: {formatted_Probs}, n_visit: {node.n_visit}, Vdone: {node.Vdone}\
            \nTreeId: {node.TreeId}, ActInfo: {formatted_ActInfo}, ActCost: {formatted_ActCost}"
        
        if node.type == 2:  # ovlp node
            fillcolor = "Magenta"
        elif node.type == 4:    # PathFnd node
            fillcolor = "lightgreen"
        elif node.n_visit == 0: # un-visit node
            fillcolor = "white"
        else:
            fillcolor = "lightblue"  # Default color

        dot.node(str(id(node)), label, shape="box", style="filled", fillcolor=fillcolor)

        for child in node.children:
            dot.edge(str(id(node)), str(id(child)))
            self.add_nodes(child, dot)


# -------------------------- Test --------------------------
# import MctsBiDir

# MT = MctsBiDir.MctsBiDirTree()
# MT.Simulate()
# MT.VisTree(MT.RootMctsBiDirNode)