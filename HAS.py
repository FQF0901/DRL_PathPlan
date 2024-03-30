import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils
import numpy as np
import ParaCfg

# 定义网格大小和边长
grid_num = ParaCfg.HASParam.grid_num
cell_size = ParaCfg.HASParam.cell_size

def get_grid_index(x, y):
    grid_x = int((x - -20) / cell_size)
    grid_y = int((y - -20) / cell_size)
    return grid_x + grid_y * grid_num

def heuristic(node, goal):
    return math.sqrt((node.x - goal.x)**2 + (node.y - goal.y)**2)

def hybrid_a_star(start, goal, obstacles):
    open_list = [start]
    closed_list = []
    current_nodes = []
    AstarPath = []
    RSpath = []
    grid_cells = [[] for _ in range(grid_num ** 2)]

    cnt = 0
    while open_list and cnt < ParaCfg.HASParam.maxEpsd + 10:    # +10 is uesed for Redundancy of EnvStep()
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
        PlanFnd, AstarPath, RSpath = utils.cal_validRS(current_node, goal, obstacles)
        if PlanFnd:
            return PlanFnd, AstarPath, RSpath, cnt

        expdNode_list = []
        expdNode_list = utils.expandNode(current_node)  # expand child nodes from curnt node

        for nodes in expdNode_list:
            nodes.h_cost = heuristic(ParaCfg.Node(nodes.x, nodes.y, nodes.theta, 0, 0), goal)

            new_idx = get_grid_index(nodes.x, nodes.y)
            if (not utils.is_overlap_node(nodes, obstacles, 0.0, 0.0)) and nodes not in grid_cells[new_idx]:
                open_list.append(nodes)
                grid_cells[new_idx].append(nodes)

        cnt = cnt + 1

    return False, [], [], cnt

def Path_show(AstarPath, RSpath, obstacles, start, goal, slot):
    fig, ax = plt.subplots()

    # 绘制障碍物
    for obstacle in obstacles:
        rect = patches.Polygon(obstacle, closed=True, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # 绘制A*路径
    if AstarPath != []:
        AstarPath_x = [node.x for node in AstarPath]
        AstarPath_y = [node.y for node in AstarPath]
        ax.plot(AstarPath_x, AstarPath_y, 'b-+')

    # 绘制RS路径
    if AstarPath != []:
        RSpath_x = RSpath.x
        RSpath_Y = RSpath.y
        ax.plot(RSpath_x, RSpath_Y, 'b-')

    # 标记起点和终点
    ax.plot(start.x, start.y, 'go', markersize=5, label='Start')
    ax.plot(goal.x, goal.y, 'ro', markersize=5, label='Goal')

    # 绘制host veh和slot
    host_veh = utils.get_Veh_corners(goal.x, goal.y, goal.theta ,0 ,0)
    host_veh_polygon = patches.Polygon(host_veh, closed=True, edgecolor='g', facecolor='none')
    ax.add_patch(host_veh_polygon)

    slot_polygon = patches.Polygon(slot, closed=True, edgecolor='b', facecolor='none')
    ax.add_patch(slot_polygon)

    xlim = (ParaCfg.HASParam.xmin, ParaCfg.HASParam.xmax)
    ylim = (ParaCfg.HASParam.ymin, ParaCfg.HASParam.ymax)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

# # 示例障碍物信息，每个障碍物用四个角点坐标表示
# obstacles = []
# corners = utils.get_rectangle_corners(7.9, 1.2, 0, 0.1, 0.1)
# obstacles.append(corners)

# start_node = ParaCfg.Node(0, 0, 0, 0, 0)
# goal_node = ParaCfg.Node(12, 5, math.pi/4, 0, 0)

# AstarPath, RSpath = hybrid_a_star(start_node, goal_node, obstacles)
# if AstarPath and RSpath:
#     print("找到路径！")
#     Path_show(AstarPath, RSpath, obstacles, start_node, goal_node)
# else:
#     print("未找到路径！")
