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
        self.exploration_weight = 10    # 这个权重待讨论
        self.expd_maxcnt = 1000 # 每个root state的MCTS tree都要充分拓展expd_maxcnt = 5000次

    def select_node(self, curt_node):   # 通过PUCT公式选择子节点中最有价值的节点
        if curt_node.type == 2: # 确保当前不是dead node
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

        return True, selected_node

    def backpropagateType(self, node):  # 除了expand_node要回溯type = 2，主程序也要用来回溯 type = 3
        # 反向传播，更新节点的 Type：充分探索分2种情况：dead 和 PathFnd
        while (node.type == 2 or node.type == 3) and node.HasNode.parent != None:   # 本节点是个dead node 或 PathFnd
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

            if utils.ChildNotVaild(new_node, EnvInfo.State):   # ovlp和RepeatMove，type = dead node
                new_node.type = 2   # type = dead node
                new_node.HasNode.g_cost = new_node.HasNode.g_cost + -40 # 该node确认是条死路,-40保证不及格
            else:
                new_node.type = 1   # type = norm node
                EnvAction = utils.EnvDRL_ActionMapping(cnt)
                new_node.HasNode.g_cost = new_node.HasNode.g_cost + utils.EnvReward(EnvAction, EnvInfo)  # 用于最后的真值Q（基于Path的评估）

            cnt = cnt + 1

            curt_node.children.append(new_node) # 都填进去，只是不选择 type = 2 的node

        for HasNode in new_HasNode_list:    # 把type回溯父节点，以免select的时候选到dead node
            _ = self.backpropagateType(HasNode)

    def backpropagateV(self, node, DnnV):
        # 反向传播，更新节点的 访问次数、累计奖励
        while node is not None:
            node.V = (node.V * node.visit_count + DnnV) / (node.visit_count + 1) # node.V = total_reward / visit_count or DnnV
            node.visit_count += 1
            node = node.parent

    def is_leaf(self, node):
        """检查是否是叶节点，即没有被扩展的节点"""
        NoChildFlag = node.children == None # 没子节点 或 子节点的P全为0？
         
        if not NoChildFlag: # 有子节点但均不vaild
            ChildNotVaildFlag = sum(child.P for child in node.children) == 0

        return NoChildFlag or ChildNotVaildFlag    # 这里还要增加判断children的P是否不为0
    
    def simulate(self, EnvInfo):  # 这是一个完整的plan流程
        node = self.root_state.MctsNode

        while True: # 探索选择，直到找到叶节点
            if self.is_leaf(node):
                LeafNode = node
                break
            node = self.select_node(node)   # PUCT: 策略 + 价值
            
        self.expand_node(LeafNode, EnvInfo)   # 这里要判断是否pathfound和openlist
        self.backpropagate(LeafNode)

        return state, action, action_probs
    
    def TkAct(self, EnvState):



# ---------------------------- Collection ---------------------------
num_episodes = 10000

env = Env.Env()

for _ in range(num_episodes):
    DRLstate, EnvState = env.reset()
    MctsTree = MCTS(EnvState)
    state_list, act_probs_list, Q_value_list = [], [], [] # state_list每个element应包含obst，SP/TP 和 【occupied grid】
    
    done = False
    while not done:
        state, action, action_probs = MctsTree.TkAct(EnvState)  # TkAct内基于该state推演了至多PlayOutOkCnt次成功规划
        next_state, reward, done, info, EnvState = env.step(action)

        state_list.append(state)
        act_probs_list.append(action_probs) # 经过PlayOutOkCnt次成功规划后action_probs应该是有可信度的
        
    MctsTree.root_state.MctsNode.V = MctsTree.root_state.MctsNodHasNode.g_cost  # 结束后给出真值用于更新DNN的Q，需要细致的评判轨迹的优劣
    MctsTree.backpropagate(MctsTree.root_state.MctsNode)

    # Q_value_list.append()

# pickle
