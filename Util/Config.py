"""
@author: Fqf
@time: 20240724
@file: Config.py
@description: shared parameters
"""

# ==================================================================
# ==================== Global Config Parameters ====================
# ==================================================================

# -------------------- Veh para --------------------
class VehParams:
    def __init__(self) -> None:
        # BX platform
        self.VehicleLength = 4.431
        self.VehicleWidth = 1.851
        self.Center2FrontAxle = 1.3895
        self.Center2RearAxle = 1.3605
        self.WheelRadius = 0.347
        self.MinTurnRadius = 5.0
        self.MaxFrntStrAng_deg = 28.78

        self.LatMargin = 0.1
        self.LgtMargin = 0.1

        # A2 platform
        # self.VehicleLength = 4.976
        # self.VehicleWidth = 2.005
        # self.Center2FrontAxle = 1.55
        # self.Center2RearAxle = 1.427
        # self.WheelRadius = 0.371
        # self.MinTurnRadius = 5.2
        # self.MaxFrntStrAng_deg = 29.98
        
        # self.LatMargin = 0.1
        # self.LgtMargin = 0.1

VehPara = VehParams()