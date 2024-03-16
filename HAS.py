import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils
import numpy as np

# 定义网格大小和边长
grid_size = 200
cell_size = 0.2

class Node:
    def __init__(self, x, y, theta, g_cost, h_cost, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.parent = parent

def get_grid_index(x, y):
    grid_x = int((x + 20) / cell_size)
    grid_y = int((y + 20) / cell_size)
    return grid_x + grid_y * grid_size

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
    grid_cells = [[] for _ in range(grid_size ** 2)]

    fig, ax = plt.subplots()

    # 绘制障碍物
    for obstacle in obstacles:
        rect = patches.Rectangle(obstacle[0], obstacle[2][0]-obstacle[0][0], obstacle[2][1]-obstacle[0][1], linewidth=1, edgecolor='r', facecolor='r')
        ax.add_patch(rect)

    cnt = 0
    while open_list and cnt < 750:
        current_node = min(open_list, key=lambda node: node.g_cost + node.h_cost)
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
                new_node = Node(new_x, new_y, new_theta, current_node.g_cost, heuristic(Node(new_x, new_y, new_theta, 0, 0), goal), current_node)

                new_idx = get_grid_index(new_node.x, new_node.y)
                if (not is_overlap(new_node, obstacles)) and new_node not in closed_list and new_node not in open_list and new_node not in grid_cells[new_idx]:
                    open_list.append(new_node)
                    grid_cells[new_idx].append(new_node)

        expanded_nodes.append((current_node.x, current_node.y))

        # 实时绘制节点
        path_x = [node[0] for node in expanded_nodes]
        path_y = [node[1] for node in expanded_nodes]
        ax.plot(path_x, path_y, 'bo', markersize=1)
        plt.pause(0.001)

        cnt = cnt + 1

    return None

def plot_path(path, obstacles, success=True):
    fig, ax = plt.subplots()

    # 绘制障碍物
    for obstacle in obstacles:
        rect = patches.Rectangle(obstacle[0], obstacle[2][0]-obstacle[0][0], obstacle[2][1]-obstacle[0][1], linewidth=1, edgecolor='r', facecolor='r')
        ax.add_patch(rect)

    if success:
        # 绘制路径
        path_x = [node.x for node in path]
        path_y = [node.y for node in path]
        ax.plot(path_x, path_y, 'g-')

    plt.xlim(-1, 6)
    plt.ylim(-1, 6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# 示例障碍物信息，每个障碍物用四个角点坐标表示
obstacles = []

for _ in range(2):
    x = np.random.uniform(4, 6)
    y = np.random.uniform(4, 6)
    yaw = np.random.uniform(0, 360)
    length, width = np.random.uniform(0.5, 1, 2)
    
    corners = utils.get_rectangle_corners(x, y, yaw, length, width)
    obstacles.append(corners)

start_node = Node(0, 0, 0, 0, 0)
goal_node = Node(10, 10, math.pi/4, 0, 0)

path = hybrid_a_star(start_node, goal_node, obstacles)
if path:
    print("找到路径！")
    plot_path(path, obstacles, success=True)
else:
    print("未找到路径！")
    # plot_path([], obstacles, success=False)
