"""
@author: Fqf
@time: 20240618
@file: Config.py
@description: Related parameters and configuration
"""

import numpy as np

# ==================================================================
# ==================== Global Config Parameters ====================
# ==================================================================

# -------------------- Veh para --------------------
class VehParams:
    def __init__(self) -> None:
        # BX platform
        # self.VehicleLength = 4.431
        # self.VehicleWidth = 1.851
        # self.Center2FrontAxle = 1.3895
        # self.Center2RearAxle = 1.3605
        # self.WheelRadius = 0.347
        # self.MinTurnRadius = 5.0
        # self.MaxFrntStrAng_deg = 28.78

        # self.LatMargin = 0.1
        # self.LgtMargin = 0.1

        # A2 platform
        self.VehicleLength = 4.976
        self.VehicleWidth = 2.005
        self.Center2FrontAxle = 1.55
        self.Center2RearAxle = 1.427
        self.WheelRadius = 0.371
        self.MinTurnRadius = 5.2
        self.MaxFrntStrAng_deg = 29.98
        
        self.LatMargin = 0.1
        self.LgtMargin = 0.1

class TreeParams:
    def __init__(self) -> None:
        self.StrActDim = 3  # need update to 7 [important]
        self.GearActDim = 2
        self.DistActDim = 1
        
        self.StrAct_list = np.linspace(-1, 1, self.StrActDim).tolist()   # Sample uniformly between -1 and 1 (StrActDim)
        self.GearAct_list = [-1, 1]
        self.ExpdStepLen_list = [0.1 * i + 0.1 for i in range(self.DistActDim)]  # Start from 0.1m and expand in steps of 0.1m

        self.TreeLvlCostCoff = 0.1
        self.SteerSpdCostCoff = 1
        self.GearShiftCostCoff = 1


VehPara = VehParams()
TreePara = TreeParams()