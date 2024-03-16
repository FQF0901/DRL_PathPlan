import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils
import numpy as np
import reeds_shepp as rs
import ParaCfg
import Env
import HAS
import utils

env = Env.Env()
# env.reset()
obj = []

start_node = HAS.Node(0, 0, 0, 0, 0)
goal_node = HAS.Node(5, 6, math.radians(-120), 0, 0)

# 计算路径
RSpath = rs.calc_optimal_path(start_node, goal_node)

corners = utils.get_rectangle_corners(4, 2, 0, 0.1, 0.1)
obj.append(corners)

for i in range(0, len(RSpath.x)):
    pathx = RSpath.x[i]
    pathy = RSpath.y[i]
    pathyaw = RSpath.yaw[i]

    RSnode = HAS.Node(pathx, pathy, pathyaw, 0, 0)
    if HAS.is_overlap(RSnode, obj):
        print('Collision')
        break
    elif i == len(RSpath.x) - 1:    # 全部RS校验完成都没有碰撞
        print('No collision')


# # 提取路径中的点坐标
x_coords = RSpath.x
y_coords = RSpath.y

# 绘制路径
plt.figure()
plt.plot(x_coords, y_coords, 'b-')
plt.plot(0, 0, 'ro')  # 起点
plt.plot(5, 5, 'go')  # 终点
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Path Visualization')
plt.show()
