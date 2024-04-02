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
    
class DQNPostProc:
    def __init__(self):
        self.doneCnt_StepCnt_list = []
        self.doneCnt_ActVehOvlp_list = []
        self.doneCnt_PathFnd_list = []
        self.doneCnt_VehOutMap_list = []

        self.donePct_StepCnt_list = []
        self.donePct_ActVehOvlp_list = []
        self.donePct_PathFnd_list = []
        self.donePct_VehOutMap_list = []

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

class HasNode:
    def __init__(self, x, y, theta, g_cost, h_cost, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.parent = parent

class MctsNode:
    def __init__(self, children=None, DnnQ = 0, DnnP = 0):
        self.HasNode = HasNode()
        self.children = children
        self.visit_count = 0   # 当前当前节点的访问次数
        self.Q = DnnQ       # 当前节点对应动作的平均动作价值
        self.P = DnnP       # DNN给出的P概率
