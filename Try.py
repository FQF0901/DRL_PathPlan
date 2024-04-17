import graphviz

class HasNode:
    def __init__(self, x=0, y=0, theta=0, g_cost=0, h_cost=0, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.parent = parent

class MctsNode:
    def __init__(self, has_node=None, children=None, visit_count=0, DnnV=0, DnnP=0, Vdone=False, Type=0):
        self.has_node = has_node if has_node is not None else HasNode()
        self.children = children if children is not None else []
        self.visit_count = visit_count   # 当前当前节点的访问次数
        self.V = DnnV       # 当前节点对应动作的平均动作价值
        self.P = DnnP       # DNN给出的P概率
        self.Vdone = Vdone  # node的V是否回溯完成
        self.type = Type    # 0:default, 1:Norm, 2:Dead, 3:PathFnd。用于记录是否充分探索
        self.idx = -1   # 和action绑定

def visualize_tree(root):
    dot = graphviz.Digraph()
    add_nodes(root, dot)
    dot.render('tree', format='png', cleanup=True)

def add_nodes(node, dot):
    dot.node(str(id(node)), f"({node.has_node.x}, {node.has_node.y})")
    for child in node.children:
        dot.edge(str(id(node)), str(id(child)))
        add_nodes(child, dot)

# 创建一个简单的 n 叉树
root = MctsNode(HasNode(1, 1), [
    MctsNode(HasNode(2, 2)),
    MctsNode(HasNode(3, 3), [
        MctsNode(HasNode(4, 4)),
        MctsNode(HasNode(5, 5)),
        MctsNode(HasNode(6, 6))
    ]),
    MctsNode(HasNode(7, 7))
])

# 可视化树
visualize_tree(root)
