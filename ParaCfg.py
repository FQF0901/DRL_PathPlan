# ======================= Global Para =======================
class VehicleParams:
    def __init__(self):
        self.length = 4.48
        self.width = 1.85
        self.rear_to_back = 1.316
        self.wheelbase = 2.75
        self.radius = 5.0

class HASParams:
    def __init__(self):
        self.xmin = -8
        self.xmax = 8
        self.ymin = -4
        self.ymax = 7
        self.grid_num = 200
        self.cell_size = 0.2
        self.step_size = 0.4
        self.maxEpsd = 300

class TrainingParams:
    def __init__(self):
        self.learning_rate = 0.1
        self.num_iterations = 2000
        self.batch_size = 500

VehPara = VehicleParams()
HASParam = HASParams()
TrainPara = TrainingParams()

# ======================= Global Type =======================

class EnvState:
    def __init__(self):
        self.ObjRect = []  # 存储随机数量obj的角点
        self.OthVehRect = []  # 存储随机数量other_veh的角点

        self.StartPntStep = []
        self.StartRectStep = None
        self.TgtPntStep = []
        self.TgtRectStep = None

class EnvInfo:
    def __init__(self):
        self.State = EnvState()
        self.VehPntInit = []
        self.VehRectInit = None
        self.SlotPntInit = []
        self.SlotRectInit = None

        self.action_z = []
        self.Reward_z = []
        self.ActionVehOvlp = False
        self.PathFnd = 0
        self.StepCnt = 0

class Node:
    def __init__(self, x, y, theta, g_cost, h_cost, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.parent = parent

class DQNPostProc:
    def __init__(self, StepCnt, ActVehOvlp, PathFnd, VehOutMap):
        self.doneCnt_StepCnt = StepCnt
        self.doneCnt_ActVehOvlp = ActVehOvlp
        self.doneCnt_PathFnd = PathFnd
        self.doneCnt_VehOutMap = VehOutMap