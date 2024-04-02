# 导入所需的库
import numpy as np
import random
import math
import utils
import ParaCfg

class MCTS:
    def __init__(self):
        # 初始化MCTS参数
        self.simulation_count = 100
        self.c_puct = 5

    def select_node(self, curt_node):
        # 通过UCT公式选择子节点中最有价值的节点
        best_value = float("-inf")
        selected_node = None
        for child_node in curt_node.children:
            exploitation_term = child_node.Q
            exploration_term = np.sqrt(child_node.HasNode.parent.n_visits) / (1 + child_node.n_visits)
            puct_value = exploitation_term + self.c_puct * child_node.P * exploration_term  # PUCT公式

            if puct_value > best_value:
                best_value = puct_value
                selected_node = child_node

        return selected_node

    def expand_node(self, curt_node):   # 这里要把不合法的动作概率全部设置为0，并补充P和Q
        new_node = ParaCfg.MctsNode
        new_HasNode_list = utils.expandNode(curt_node)

        for HasNode in new_HasNode_list:
            new_node.HasNode = HasNode
            DnnQ, DnnP = utils.DNN(new_node.HasNode)
            new_node.Q = DnnQ
            new_node.P = DnnP

            if utils.ChildNotVaild(new_node):   # ovlp和RepeatMove，P为0或不往children里写入
                new_node.P = 0  
            curt_node.children.append(new_node) # 不vaild就不想里面写入？

        # return curt_node.children

    def backpropagate(self, node, DnnQ):
        # 反向传播，更新节点的信息（访问次数、累计奖励等）
        while node is not None:
            node.Q = (node.Q * node.visit_count + DnnQ) / (node.visit_count + 1) # node.Q = total_reward / visit_count or DnnQ
            node.visit_count += 1
            node = node.parent

    def is_leaf(self, node):
        """检查是否是叶节点，即没有被扩展的节点"""
        return node.children == None    # 这里还要增加判断children的P是否不为0

    def is_fully_expanded(self):
        # 检查节点是否完全扩展
        return len(self.children) == len(self.state.get_legal_actions())
    
    def simulate(self, root_node):
        cnt = 0

        while cnt < ParaCfg.HASParam.maxEpsd and PathNotFnd and Openlist != [] :
            node = root_node

            while True:
                if self.is_leaf(node):
                    curt_node = node
                    break
                node = self.select_node(node)
                
            self.expand_node(curt_node)   # 这里要判断是否pathfound和openlist
            self.backpropagate(curt_node)
            cnt = cnt + 1

        node.Q = TrueValue  # 如果终止了，就应该给出真值用于更新DNN的Q
        self.backpropagate(node)
    
    def search(self):
        # 主循环执行路径规划过程
        for _ in range(num_iterations):
            # 使用MCTS进行搜索
            current_node = initial_node
            for _ in range(self.simulation_count):
                selected_node = self.select_node(current_node)
                if selected_node is not self.is_fully_expanded:
                    new_node = self.expand_node(selected_node)
                    reward = self.simulate(new_node)
                    self.backpropagate(new_node, reward)
                else:
                    reward = self.simulate(selected_node)
                    self.backpropagate(selected_node, reward)

            # 结合MCTS和DQN结果
            action = self.get_best_action()  # 从MCTS中获取最佳行动


num_iterations = 10
initial_node = ParaCfg.MctsNode()


