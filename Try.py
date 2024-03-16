import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils
import numpy as np
import reeds_shepp as rs
import numpy as np
import Env
# import HAS
import ParaCfg

env = Env.Env()
env.reset()

# calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=STEP_SIZE)
path = rs.calc_all_paths(env.SP[0], env.SP[1], env.SP[2], env.TP[0], env.TP[1], env.TP[2], ParaCfg.VehPara.radius, 0.2)

