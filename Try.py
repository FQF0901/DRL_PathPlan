class HasNode:
    def __init__(self, x = 0, y = 0, theta = 0, g_cost = 0, h_cost = 0, parent = None):
        self.x = x
        self.y = y
        self.theta = theta
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.parent = parent

node = HasNode()

node.y
node.parent