import numpy as np
import Env
import HAS
import ParaCfg

# Env
env = Env.Env()
env.reset()
# env.show()    # Need to confirm the consistency between Env sending and HAS receiving

# HAS
start_node = ParaCfg.Node(env.EnvInfo.SlotPntInit[0], env.EnvInfo.SlotPntInit[1], env.EnvInfo.SlotPntInit[2], 0, 0)
goal_node = ParaCfg.Node(env.EnvInfo.VehPntInit[0], env.EnvInfo.VehPntInit[1], env.EnvInfo.VehPntInit[2], 0, 0)
obstacles = env.EnvInfo.State.ObjRect + env.EnvInfo.State.OthVehRect

PlanFlag, AstarPath, RSpath = HAS.hybrid_a_star(start_node, goal_node, obstacles)

# Visualization
if PlanFlag:
    print("找到路径！")
else:
    print("未找到路径！")
    
HAS.Path_show(AstarPath, RSpath, obstacles, start_node, goal_node, env.EnvInfo.SlotRectInit)
    