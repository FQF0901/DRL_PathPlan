"""
@author: Fqf
@time: 20240618
@file: Env.py 
@description: Sim env, more suitable for simulating real-time plan(PPO), not suitable for simulating MCTS
"""

import GlbVar
import DrlUtil
from GlbVar import thread_local

# ==========================================================
# ======================== Env Info ========================
# ==========================================================
class Env:
    def __init__(self) -> None:
        self.PcptInfo = thread_local.PcptInfo
        self.Reward = 0
        self.CycleCnt = 0 
    
    def reset(self):
        self.PcptInfo = ([], [], [], []) # Only here can write into PcptGeo
        self.Reward = 0
        self.CycleCnt = 0

    def step(self, action):
        next_state = DrlUtil.PrdtVehPoseBicyMod(self.CrntPt, action[1], action[2], action[3])  # action : StrAngRate, gear, dist
        reward = 0  # need to update [important]
        done = False    # need to update [important]
        info = []

        return next_state, reward, done, info

    # def show(self):