"""
@author: Fqf
@time: 20240618
@file: DrlCfg.py
@description: Related parameters and configuration
"""

import numpy as np

# ==================================================================
# ==================== Global Config Parameters ====================
# ==================================================================

class TreeParams:
    def __init__(self) -> None:
        self.StrActDim = 3  # need update to 7 [important]
        self.GearActDim = 2
        self.DistActDim = 1
        
        self.StrAct_list = np.linspace(-1, 1, self.StrActDim).tolist()   # Sample uniformly between -1 and 1 (StrActDim)
        self.GearAct_list = [-1, 1]
        self.ExpdStepLen_list = [0.2 * i + 0.1 for i in range(self.DistActDim)]  # Start from 0.1m and expand in steps of 0.1m

        self.TreeLvlCostCoff = 0.1
        self.SteerSpdCostCoff = 1
        self.GearShiftCostCoff = 1
        self.ExpdNodeCost = 0.2

        self.TreeType = 1   # 1: Mcts, 2: Mcts efficient, 3: Mcts Bi-direction


TreePara = TreeParams()