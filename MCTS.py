# 导入所需的库
import numpy as np
import random
import torch
import utils
import ParaCfg
import Env
import logging
import tqdm
import collections

# ---------------------------- MCTS Tree ---------------------------
class MCTS:
    def __init__(self, EnvState):
        self.root_state = ParaCfg.MctsState(EnvState = EnvState)
        self.exploration_weight = 100    # 这个权重待讨论
        self.gamma = 0.97   # state value回溯时的衰减   0.95^10=0.598, 0.97^10=0.737
        self.expd_maxcnt = 1000 # 每个root state的MCTS tree都要充分拓展expd_maxcnt = 5000次

    def select_node(self, curt_node):   # 通过PUCT公式选择子节点中最有价值的节点。不能用min heap，因为没有随机性
        if curt_node.type == 2 or curt_node.type == 3: # 确保当前不是dead node
            return False, curt_node

        best_value = float("-inf")
        selected_node = None
        for child_node in curt_node.children:
            exploitation_term = child_node.V
            exploration_term = np.sqrt(child_node.HasNode.parent.visit_count) / (1 + child_node.visit_count)
            puct_value = exploitation_term + self.exploration_weight * child_node.P * exploration_term  # PUCT公式

            if puct_value > best_value and child_node.type != 2 and child_node.type != 3:   # type = 2有两种可能：ovlp和repeat move
                best_value = puct_value
                selected_node = child_node

            selected_node.visit_count = selected_node.visit_count + 1

        return True, selected_node

    def backpropagate_Type(self, node):  # 除了expand_node要回溯type = 2，主程序也要用来回溯 type = 3
        # 反向传播，更新节点的 Type：充分探索分2种情况：dead 和 PathFnd
        while node is not None and (node.type == 2 or node.type == 3):   # 本节点是个dead node 或 PathFnd
            nodeBro_list = node.HasNode.parent.children # 找本node的所有兄弟节点
            nodeBro_PathFnd = False

            for nodeBro in nodeBro_list:
                if nodeBro.type != 2 and nodeBro.type != 3:
                    return True   # 存在 非dead node且 非 PathFnd， 则回溯完成并退出
                if nodeBro.type == 3:
                    nodeBro_PathFnd = True

            node.HasNode.parent.type = 3 if nodeBro_PathFnd else 2    # 兄弟节点均为dead node，则父节点也应为dead node
            node = node.HasNode.parent

    def expand_node(self, curt_node, EnvInfo):   # 这里要把不合法的动作type置2，这个type需要回溯父节点
        new_node = ParaCfg.MctsNode()
        new_HasNode_list = utils.expandNode(curt_node)
        cnt = 0

        for HasNode in new_HasNode_list:
            new_node.HasNode = HasNode
            DnnV, DnnP = utils.DNN(new_node.HasNode)    # 需要补充DNN
            new_node.V = DnnV   # 用于指导select
            new_node.P = DnnP   # 用于指导select
            new_node.idx = cnt

            if utils.ChildNotVaild(curt_node.idx, new_node, EnvInfo.State):   # ovlp和RepeatMove，type = dead node
                new_node.type = 2   # type = dead node
                # new_node.HasNode.g_cost = new_node.HasNode.g_cost # ovlp和repear move不建议惩罚
            else:
                new_node.type = 1   # type = norm node
                new_node.HasNode.g_cost = new_node.HasNode.g_cost + utils.MctsExpdrRwd(new_node, EnvInfo)  # utils.EnvReward无需判断PathFnd

            cnt = cnt + 1

            curt_node.children.append(new_node) # 都填进去，只是不选择 type = 2 的node

        return new_HasNode_list
    
    def is_leaf(self, node):
        """检查是否是叶节点，即没有被扩展的节点"""
        return node.children == None
    
    def backpropagateV(self, node):
        if node.visit_count == 1:  # 这里是递归的尽头, =1是selected once node
            node.V = node.HasNode.g_cost + 100 if node.type == 3 else 0
            node.Vdone = True
            return
        elif node.visit_count == 0: # =0是leaf node
            node.V = node.HasNode.g_cost
            node.Vdone = True
            return

        # 遍历所有子节点
        for child in node.children:
            # 如果子节点的Vdone为False，则先更新该子节点的Vdone
            if not child.Vdone:
                self.backpropagateV(child)

        # 检查所有子节点的Vdone是否均为True
        all_children_done = all(child.Vdone for child in node.children)

        if all_children_done:
            # 计算父节点的DnnV
            node.V = sum(child.V * child.visit_count / (node.visit_count - 1) for child in node.children)
   
    
    def simulate(self, EnvInfo):  # 这是针对某个init state的一个完整充分的探索流程
        node = self.root_state.MctsNode

        for cnt in range(self.expd_maxcnt):   # 充分拓展self.expd_maxcnt次 或 OpenList = []

            while True: # 探索选择，直到找到叶节点
                if self.is_leaf(node):
                    LeafNode = node
                    break

                OpListFlg, node = self.select_node(node)   # 选择该node的children，更新n_visit，并判断root_node是否openlist = []
                if OpListFlg == 0:
                    print('Episode end due to OpenList = [] !')
                    return 2, cnt
            
            # 还要判断LeafNode是否可以PathFnd
            obstacles = EnvInfo.State.ObjRect + EnvInfo.State.OthVehRect
            PlanFnd, _, _ = utils.cal_validRS(LeafNode, EnvInfo.VehPntInit, obstacles)
            if PlanFnd:
                LeafNode.type = 3
                # LeafNode.V = LeafNode.HasNode.g_cost + 100
                _ = self.backpropagate_Type(LeafNode)
                
            new_Node_list = self.expand_node(LeafNode, EnvInfo)   # 仅判断是否ovlp和repeat move
            for newNode in new_Node_list:    # 把type回溯父节点，以免select的时候选到dead node或PathFnd
                _ = self.backpropagate_Type(newNode)

        self.backpropagateV(node)    # 1000次充分探索后要回溯state value
        
        return 1, self.expd_maxcnt
    
    def StoreTreeInfo(self, node, state_list, act_probs_list, V_value_list):

        if node.visit_count == 1:  # 这里是store时的递归尽头, =1是selected once node
            return

        # 遍历所有子节点
        for child in node.children:
            self.backpropagateV(child)
    
# ---------------------------- Collection ---------------------------
num_episodes = 10000

env = Env.Env()

for _ in range(num_episodes):
    DRLstate, EnvState = env.reset()
    MctsTree = MCTS(EnvState)
    state_list, act_probs_list, V_value_list = [], [], [] # state_list每个element应包含obst，SP/TP 和 【occupied grid】
        
    DoneFlag, expd_cnt = MctsTree.simulate(EnvInfo)

    MctsTree.StoreTreeInfo(MctsTree.root_state.MctsNode, state_list, act_probs_list, V_value_list)
    
    # pickle
