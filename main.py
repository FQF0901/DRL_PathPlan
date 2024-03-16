import numpy as np
import Env
import HAS

env = Env.Env()
env.reset()
env.show()

# 沿着行方向（垂直方向）拼接两个数组

start_node = HAS.Node(env.TP[0], env.TP[1], env.TP[2], 0, 0)
goal_node = HAS.Node(env.SP[0], env.SP[1], env.SP[2], 0, 0)
obstacles = env.obj + env.other_veh

path = HAS.hybrid_a_star(start_node, goal_node, obstacles)

if path:
    print("找到路径！")
    HAS.Path_show(path, obstacles, start_node, goal_node)
else:
    print("未找到路径！")