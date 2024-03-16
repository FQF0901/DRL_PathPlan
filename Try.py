import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils
import numpy as np
import reeds_shepp as rs
import ParaCfg
import numpy as np
import Env

# env = Env.Env()
# env.reset()

# 计算路径
path = rs.calc_optimal_path(0, 0, 0, 5, 5, math.radians(-120), 0.2, 0.02)


# # 提取路径中的点坐标
x_coords = path.x
y_coords = path.y

# 绘制路径
plt.figure()
plt.plot(x_coords, y_coords, 'b-')
plt.plot(0, 0, 'ro')  # 起点
plt.plot(5, 5, 'go')  # 终点
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Path Visualization')
plt.show()
