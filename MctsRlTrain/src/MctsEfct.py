"""
@author: Fqf
@time: 20240731
@file: MctsEfficient.py
@description: Decision Tree with max heap or list to optimize CPU efficiency
"""

import os
from collections import deque
import graphviz
import GlbVar
import DrlUtil
import random
import DrlCfg
import matplotlib.cm as cm
import matplotlib.colors as colors

global PcptInfo

# ==========================================================
# ===================== MctsEfct Tree ======================
# ==========================================================
class MctsEfctTree:
    def __init__(self) -> None:
        # 1. Config
        self.exploration_weight = 10
        self.gamma = 0.99
        self.beta = 0.6
        self.epsilon = 0

        # 2. Single-directional expansion tree with interchangeable start and target node
        self.RootMctsEfctNode = GlbVar.MctsEfctNode()
        self.OpenDeque = deque()
        self.OpenDeque.append(self.RootMctsEfctNode)
        self.TargetPose = GlbVar.Node()

# ----------------------- Select Node ----------------------
    """ Select node(type should equal to 1) in openlist """
    def SelectNode(self):

        if random.random() < self.epsilon:  # select node by value, need update [important]
            pass
        else:   # select node randomly
            idx = random.randint(0, len(self.OpenDeque)-1)
            selected_node = self.OpenDeque[idx]
            selected_node.n_visit = 1
            if not selected_node.parent == None:
                selected_node.parent.n_visit = 2
            
            self.OpenDeque.remove(selected_node)
        
        return selected_node

# ----------------------- Expand Node ----------------------
    """Expand the node and init the related info"""
    def ExpandNode(self, crnt_node):
        # 1. Expand Node by bicyc-model
        new_node_list, ExpdAct_list = DrlUtil.Expd_Node(crnt_node.node)

        # 2. Add element info into expanded node
        for index, node in enumerate(new_node_list, start=1):
            NewMctsNode = GlbVar.MctsEfctNode()

            # 2.1 New Mcts Node Element Info
            NewMctsNode.node = node
            NewMctsNode.parent = crnt_node
            DnnV = random.uniform(0, 100)    # Need delete [important]
            NewMctsNode.Value = DnnV    # Need update [important]
            NewMctsNode.TreeLvl = crnt_node.TreeLvl + 1

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
            if DrlUtil.IsOvlpAllObst(node):
                NewMctsNode.type = 2    # 2:Ovlp
            elif DrlUtil.ChkRepeatAct(NewMctsNode):
                NewMctsNode.type = 3    # 3:RepeatMove (Impossible that collision and repeat both satisfied)
            else:
                NewMctsNode.type = 1    # 1:Unexplored
                self.OpenDeque.append(NewMctsNode)  # Push node into openlist

            # 3. return expand node list
            crnt_node.children.append(NewMctsNode)

# ---------------------- Backpropagate ---------------------   
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
            node_avrg_value = 0 
            node_max_value = 0 
            node_n_visit = 1

            for child_node in node.children:
                node_n_visit = node_n_visit + child_node.n_visit    # To avoid the loss of value in backpropagation

            # Whether to use maxpooling or averagepooling requires careful consideration [important]
            for child_node in node.children:
                itmdt_reward = 0    # child_node.ActCost
                Probs = 1 / (DrlCfg.TreePara.DistActDim * DrlCfg.TreePara.GearActDim * DrlCfg.TreePara.StrActDim)  # child_node.n_visit / (node_n_visit - 1). To avoid the loss of value in backpropagation

                node_avrg_value = node_avrg_value + (itmdt_reward + self.gamma * child_node.Value) * Probs    # Averagepooling-like solution

                if child_node.Value > node_max_value / self.beta:  # Maxpooling-like solution
                    node_max_value = self.beta * child_node.Value

            node.Value = max(0, min(0.5 * (node_avrg_value + node_max_value), 100))
            node.n_visit = node_n_visit
            node.Vdone = True
    
# ----------------------- Simulation -----------------------
    """This is a complete exploration process for root node"""
    def Simulate(self, expd_maxcnt=30):

        for cnt in range(expd_maxcnt):

            # 1. Select Node
            # 1.1 Explore and select until a leaf node is found
            if len(self.OpenDeque) > 0:
                SelectedLeafNode = self.SelectNode()    # select node whose type = 1
            else:
                print('-- Episode end due to  openlist = [] !')
                break

            PathFnd, _ = DrlUtil.CalValidRS(self.TargetPose, SelectedLeafNode.node)
            if PathFnd:
                SelectedLeafNode.type = 4
            else:
                # 2. Expand Node based on leaf node, but if a branch is PathFnd, then will not explore this branch
                self.ExpandNode(SelectedLeafNode)   

        _ = self.BackpropagateValue(self.RootMctsEfctNode)

        if cnt >= expd_maxcnt-1:
            print('-- Episode end due to reach expd_maxcnt !')

        return cnt

# ---------------------- StoreTreeInfo ---------------------
    """Store the tree information after backpropagate value"""
    # External packaging interface
    def StoreTreeInfo(self):
        state_list, V_value_list = [], []
        self.TravslTreeInfo(self.RootMctsEfctNode, state_list, V_value_list)

        return state_list, V_value_list

    # Recursively traverse the entire tree
    def TravslTreeInfo(self, MctsNode, state_list, V_value_list):

        for child_node in MctsNode.children:
            if child_node.n_visit > 1 and child_node.StoreDone == False:
                
                state_list.append([child_node.node, self.TargetPose, PcptInfo])
                V_value_list.append([GrandChild.Value for GrandChild in child_node.children])

                child_node.StoreDone = True

                self.TravslTreeInfo(child_node, state_list, V_value_list)
            
# ---------------------- Visualization ---------------------
    def VisTree(self, scene_pkl_file='', time_slice_idx='', store_path=os.getcwd()):
        root = self.RootMctsEfctNode

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
        formatted_ActCost = "{:.2f}".format(node.ActCost)
        
        label = f"({formatted_x}, {formatted_y}, {formatted_yaw_rad})\
            \nValue: {formatted_V}, n_visit: {node.n_visit}, Type: {node.type}, Vdone: {node.Vdone}\
            \nTreeLvl: {node.TreeLvl}, ActInfo: {formatted_ActInfo}, ActCost: {formatted_ActCost}"
        
        if node.type == 2:  # ovlp node
            fillcolor = "Magenta"
        elif node.type == 4:    # PathFnd node
            fillcolor = "lightgreen"
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
        norm = colors.Normalize(vmin=-20, vmax=120)
        
        rgba_color = cmap(norm(value))  # Map value to a color in the colormap
        hex_color = colors.rgb2hex(rgba_color)  # Convert RGBA tuple to hexadecimal color code

        return hex_color


# -------------------------- Test --------------------------
# import Mcts

# MtEfct = MctsEfctTree()
# MtEfct.Simulate()
# MtEfct.VisTree(MtEfct.RootMctsEfctNode)