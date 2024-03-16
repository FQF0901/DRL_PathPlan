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

def heuristic(node, goal):
    return math.sqrt((node.x - goal.x)**2 + (node.y - goal.y)**2)

def is_valid_node(node, obstacles):
    host_veh = utils.get_Veh_corners(node.x, node.y, node.theta)

    for obstacle in obstacles:
        if utils.check_overlap(obstacle, host_veh):
            return False
    return True

def hybrid_a_star(start, goal, obstacles):
    open_list = [start]
    closed_list = []

    cnt = 0
    while open_list and cnt < 3000:
        current_node = min(open_list, key=lambda node: node.g_cost + node.h_cost)
        open_list.remove(current_node)
        closed_list.append(current_node)

        if abs(current_node.x - goal.x) < 0.1 and abs(current_node.y - goal.y) < 0.1:
            path = []
            while current_node:
                path.append(current_node)
                current_node = current_node.parent
            return path[::-1]

        for steering_angle in [-30, 0, 30]:
            for gear in [-1, 1]:
                new_x, new_y, new_theta = utils.cal_VechPose(current_node.x, current_node.y, current_node.theta, steering_angle, gear, 0.2)
                new_node = Node(new_x, new_y, new_theta, current_node.g_cost + 1, heuristic(Node(new_x, new_y, new_theta, 0, 0), goal), current_node)

            if is_valid_node(new_node, obstacles) and new_node not in closed_list:
                if new_node not in open_list:
                    open_list.append(new_node)
        
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
# obstacles = [
#     [(10, 1), (10, 2), (20, 2), (20, 1)],
#     [(30, 3), (30, 4), (40, 4), (40, 3)]
# ]
obstacles = []

for _ in range(2):
            x = np.random.uniform(15, 20)
            y = np.random.uniform(30, 40)
            yaw = np.random.uniform(0, 360)
            length, width = np.random.uniform(0.1, 0.5, 2)
            
            corners = utils.get_rectangle_corners(x, y, yaw, length, width)
            obstacles.append(corners)

start_node = Node(0, 0, 0, 0, 0)
goal_node = Node(50, 50, math.pi/4, 0, 0)

path = hybrid_a_star(start_node, goal_node, obstacles)
if path:
    print("找到路径！")
    plot_path(path, obstacles, success=True)
else:
    print("未找到路径！")
    plot_path([], obstacles, success=False)
