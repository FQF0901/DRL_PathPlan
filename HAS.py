import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils
import numpy as np
import reeds_shepp as rs
import ParaCfg

# 定义网格大小和边长
grid_num = ParaCfg.MapParam.grid_num
cell_size = ParaCfg.MapParam.cell_size

class Node:
    def __init__(self, x, y, theta, g_cost, h_cost, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.parent = parent

def get_grid_index(x, y):
    grid_x = int((x - -20) / cell_size)
    grid_y = int((y - -20) / cell_size)
    return grid_x + grid_y * grid_num

def heuristic(node, goal):
    return math.sqrt((node.x - goal.x)**2 + (node.y - goal.y)**2)

def is_overlap(node, obstacles):
    host_veh = utils.get_Veh_corners(node.x, node.y, node.theta)

    for obstacle in obstacles:
        if utils.check_overlap(obstacle, host_veh):
            return True
    return False

def hybrid_a_star(start, goal, obstacles):
    open_list = [start]
    closed_list = []
    expanded_nodes = []
    current_nodes = []
    grid_cells = [[] for _ in range(grid_num ** 2)]

    cnt = 0
    while open_list and cnt < 750:
        current_node = min(open_list, key=lambda node: node.g_cost + node.h_cost)
        current_nodes.append(current_node)  # All selected nodes
        open_list.remove(current_node)

        current_idx = get_grid_index(current_node.x, current_node.y)
        if current_node in open_list:
            open_list.remove(current_node)
        elif current_node in closed_list:
            closed_list.remove(current_node)
        grid_cells[current_idx].append(current_node)

        closed_list.append(current_node)

        if abs(current_node.x - goal.x) < 0.1 and abs(current_node.y - goal.y) < 0.1:
            path = []
            while current_node:
                path.append(current_node)
                current_node = current_node.parent
            return path[::-1]

        for steering_angle in [-1, 0, 1]:
            for gear in [-1, 1]:
                new_x, new_y, new_theta = utils.cal_VechPose(current_node.x, current_node.y, current_node.theta, steering_angle, gear, 0.4)
                new_node = Node(new_x, new_y, new_theta, current_node.g_cost + 0.02, heuristic(Node(new_x, new_y, new_theta, 0, 0), goal), current_node)

                new_idx = get_grid_index(new_node.x, new_node.y)
                if (not is_overlap(new_node, obstacles)) and new_node not in grid_cells[new_idx]:
                    open_list.append(new_node)
                    grid_cells[new_idx].append(new_node)

                    expanded_nodes.append((new_node.x, new_node.y)) # All expand nodes

        cnt = cnt + 1

    return None

def Path_show(path, obstacles, start, goal):
    fig, ax = plt.subplots()

    # 绘制障碍物
    for obstacle in obstacles:
        rect = patches.Polygon(obstacle, closed=True, linewidth=1, edgecolor='r', facecolor='r')
        ax.add_patch(rect)

    # 绘制路径
    path_x = [node.x for node in path]
    path_y = [node.y for node in path]
    ax.plot(path_x, path_y, 'b-')

    # 标记起点和终点
    ax.plot(start.x, start.y, 'go', markersize=10, label='Start')
    ax.plot(goal.x, goal.y, 'ro', markersize=10, label='Goal')

    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

# # 示例障碍物信息，每个障碍物用四个角点坐标表示
# obstacles = []
# corners = utils.get_rectangle_corners(7.9, 1.2, 0, 0.1, 0.1)
# obstacles.append(corners)

# start_node = Node(0, 0, 0, 0, 0)
# goal_node = Node(12, 5, math.pi/4, 0, 0)

# path = hybrid_a_star(start_node, goal_node, obstacles)
# if path:
#     print("找到路径！")
#     Path_show(path, obstacles, start_node, goal_node)
# else:
#     print("未找到路径！")
