import numpy as np
import matplotlib.pyplot as plt
import Env
import HAS

# 创建一个包含两个子图的画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Env
env = Env.Env()
env.reset()
ax1.set_title('Env')
env.show()

# HAS
start_node = HAS.Node(env.TP[0], env.TP[1], env.TP[2], 0, 0)
goal_node = HAS.Node(env.SP[0], env.SP[1], env.SP[2], 0, 0)
obstacles = env.obj + env.other_veh

path, RSpath = HAS.hybrid_a_star(start_node, goal_node, obstacles)
if path and RSpath:
    print("找到路径！")
    ax2.set_title('HAS Path')
    HAS.Path_show(path, RSpath, obstacles, start_node, goal_node)
else:
    print("未找到路径！")

plt.tight_layout()
plt.show()
