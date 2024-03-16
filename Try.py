import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils
import numpy as np

class Node:
    def __init__(self, x, y, theta, g_cost, h_cost, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.parent = parent

start_node = Node(1, 2, math.radians(45), 0, 0)
open_list = [start_node]
current_node = min(open_list, key=lambda node: node.g_cost + node.h_cost)

expanded_nodes = []

# 创建图形和坐标轴
fig, ax = plt.subplots()

# 绘制起始节点
ax.plot(start_node.x, start_node.y, 'ro', markersize=5, label='Start Node')

for steer in [-1, 0, 1]:
    for gear in [-1, 1]:
        new_x, new_y, new_theta = utils.cal_VechPose(current_node.x, current_node.y, current_node.theta, steer, gear, 4)
        
        # 绘制新节点
        expanded_nodes.append((new_x, new_y))
        
# 绘制扩展节点
for node in expanded_nodes:
    ax.plot(node[0], node[1], 'bo', markersize=3, label='Expanded Node')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Expanded Nodes Visualization')
plt.legend()
ax.grid(True)
ax.set_aspect('equal', adjustable='box')
plt.show()