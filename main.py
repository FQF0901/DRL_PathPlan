import numpy as np
import Env
import HAS

env = Env.Env()
env.reset()
env.show()

# x, y = 51, 31
# sx, sy, syaw0 = env.SP[0], env.SP[1], np.deg2rad(env.SP[2])
# gx, gy, gyaw0 = env.TP[0], env.TP[1], np.deg2rad(env.TP[2])

# env.obj
# env.other_veh
# ox, oy = design_obstacles(x, y)

# t0 = time.time()
# path = hybrid_astar_planning(sx, sy, syaw0, gx, gy, gyaw0,
#                                 ox, oy, C.XY_RESO, C.YAW_RESO)