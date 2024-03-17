import numpy as np
import Env
import HAS

# Env
env = Env.Env()
env.reset()
env.show()

# HAS
start_node = HAS.Node(env.TP[0], env.TP[1], env.TP[2], 0, 0)
goal_node = HAS.Node(env.SP[0], env.SP[1], env.SP[2], 0, 0)
obstacles = env.obj + env.other_veh

path, RSpath = HAS.hybrid_a_star(start_node, goal_node, obstacles)
if path and RSpath:
    print("找到路径！")
    HAS.Path_show(path, RSpath, obstacles, start_node, goal_node)
else:
    print("未找到路径！")
    