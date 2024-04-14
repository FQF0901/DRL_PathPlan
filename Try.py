class MctsNode:
    def __init__(self, children=None, visit_count=0, DnnV=0, DnnP=0, Vdone=False, Type=0):
        self.children = children
        self.visit_count = visit_count   # 当前节点的访问次数
        self.DnnV = DnnV       # 当前节点对应动作的平均动作价值
        self.DnnP = DnnP       # DNN给出的P概率
        self.Vdone = Vdone  # node的V是否回溯完成
        self.type = Type    # 0:default, 1:Norm, 2:Dead, 3:PathFnd。用于记录是否充分探索

    def backpropagateV(self, node):
        if not node.children:
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
            total_child_v = sum(child.DnnV * child.visit_count for child in node.children)
            total_visits = sum(child.visit_count for child in node.children)
            if total_visits != 0:
                node.DnnV = total_child_v / total_visits
            node.Vdone = True

# 创建节点
A = MctsNode()
B = MctsNode()
C = MctsNode()
D = MctsNode()
E = MctsNode()
F = MctsNode()
G = MctsNode()

# 构建二叉树结构
A.children = [B, C]
B.children = [D, E]
C.children = [F, G]

# 给节点添加虚拟的DnnV和visit_count值
D.DnnV = 2
E.DnnV = 3
F.DnnV = 4
G.DnnV = 1

D.Vdone = 1
E.Vdone = 1
F.Vdone = 1
G.Vdone = 1

A.visit_count = 10
B.visit_count = 6
C.visit_count = 4
D.visit_count = 4
E.visit_count = 2
F.visit_count = 3
G.visit_count = 1

# 执行回溯计算DnnV
A.backpropagateV(A)

# 打印结果
print("Node A DnnV:", A.DnnV)
print("Node B DnnV:", B.DnnV)
print("Node C DnnV:", C.DnnV)
