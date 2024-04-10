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
        self.c_puct = 5
        self.grid_cells = [[] for _ in range(ParaCfg.HASParam.grid_num ** 2)]   # used for check repeat state
        self.root_state = ParaCfg.MctsState(EnvState = EnvState)

    def select_node(self, curt_node):
        # 通过PUCT公式选择子节点中最有价值的节点
        best_value = float("-inf")
        selected_node = None
        for child_node in curt_node.children:
            exploitation_term = child_node.Q
            exploration_term = np.sqrt(child_node.HasNode.parent.visit_count) / (1 + child_node.visit_count)
            puct_value = exploitation_term + self.c_puct * child_node.P * exploration_term  # PUCT公式

            if puct_value > best_value and child_node.P != 0:
                best_value = puct_value
                selected_node = child_node
        
        if selected_node == None:  # 如何所有child都不vaild，则该node不应该选择
            curt_node.P = 0
            # curt_node.Q = 0 # 不能给float("-inf")，太小在回溯时会过于影响父节点。干脆不给人工值
        else:
            curt_idx = utils.get_grid_index(curt_node.x, curt_node.y)   # used for check repeat state
            self.grid_cells[curt_idx].append(curt_node)

        return selected_node

    def expand_node(self, curt_node, EnvState):   # 这里要把不合法的动作概率全部设置为0，并补充P和Q
        new_node = ParaCfg.MctsNode()
        new_HasNode_list = utils.expandNode(curt_node)
        cnt = 0

        for HasNode in new_HasNode_list:
            new_node.HasNode = HasNode
            DnnQ, DnnP = utils.DNN(new_node.HasNode)    # 需要补充DNN
            new_node.Q = DnnQ
            new_node.P = DnnP

            EnvAction = utils.EnvDRL_ActionMapping(cnt)
            # new_node.HasNode.g_cost = new_node.HasNode.g_cost + utils.EnvReward(EnvAction, EnvInfo)  # 用于最后的真值Q（基于Path的评估）
            cnt = cnt + 1

            if utils.ChildNotVaild(new_node, EnvState, self.grid_cells):   # ovlp和RepeatMove，P为0
                new_node.P = 0  
                # new_node.Q = 0 # 不能给float("-inf")，太小在回溯时会过于影响父节点。干脆不给人工值

            curt_node.children.append(new_node) # 都填进去，只是不vaild的不选择

    def backpropagate(self, node, DnnQ):
        # 反向传播，更新节点的信息（访问次数、累计奖励等）
        while node is not None:
            node.Q = (node.Q * node.visit_count + DnnQ) / (node.visit_count + 1) # node.Q = total_reward / visit_count or DnnQ
            node.visit_count += 1
            node = node.parent

    def is_leaf(self, node):
        """检查是否是叶节点，即没有被扩展的节点"""
        NoChildFlag = node.children == None # 没子节点
         
        if not NoChildFlag: # 有子节点但均不vaild
            ChildNotVaildFlag = sum(child.P for child in node.children) == 0

        return NoChildFlag or ChildNotVaildFlag    # 这里还要增加判断children的P是否不为0
    
    def simulate(self):  # 这是一个完整的plan流程
        node = self.root_state.MctsNode

        while True: # 探索选择，直到找到叶节点
            if self.is_leaf(node):
                LeafNode = node
                break
            node = self.select_node(node)   # PUCT: 策略 + 价值
            
        self.expand_node(LeafNode, EnvState)   # 这里要判断是否pathfound和openlist
        self.backpropagate(LeafNode)

        return state, action, action_probs

# ---------------------------- Collection ---------------------------
num_episodes = 10000

env = Env.Env()

for _ in range(num_episodes):
    DRLstate, EnvState = env.reset()
    MctsTree = MCTS(EnvState)
    state_list, act_probs_list, Q_value_list = [], [], [] # state_list每个element应包含obst，SP/TP 和 【occupied grid】
    
    done = False
    while not done:
        state, action, action_probs = MctsTree.simulate()  # 用于policy net训练
        next_state, reward, done, info, EnvState = env.step(action)

        state_list.append(state)
        act_probs_list.append(action_probs)
        
    MctsTree.root_state.MctsNode.Q = MctsTree.root_state.MctsNodHasNode.g_cost  # 结束后给出真值用于更新DNN的Q，需要细致的评判轨迹的优劣
    MctsTree.backpropagate(MctsTree.root_state.MctsNode)

    # Q_value_list.append()

# pickle
