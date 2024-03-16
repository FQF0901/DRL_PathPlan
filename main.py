import Env
import HAS

env =Env.env
env.reset()

print(env.SP)

# x, y = 51, 31
# sx, sy, syaw0 = 10.0, 7.0, np.deg2rad(120.0)
# gx, gy, gyaw0 = 45.0, 20.0, np.deg2rad(90.0)

# ox, oy = design_obstacles(x, y)

# t0 = time.time()
# path = hybrid_astar_planning(sx, sy, syaw0, gx, gy, gyaw0,
#                                 ox, oy, C.XY_RESO, C.YAW_RESO)