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
    def __init__(self, x = 0, y = 0, theta = 0, g_cost = 0, h_cost = 0, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.parent = parent

class MctsNode:
    def __init__(self,children=None, visit_count = 0, DnnV = 0, DnnP = 0, Vdone = False, Type = 0):
        self.HasNode = HasNode()
        self.children = children
        self.visit_count = visit_count   # 当前当前节点的访问次数
        self.V = DnnV       # 当前节点对应动作的平均动作价值
        self.P = DnnP       # DNN给出的P概率
        self.Vdone = Vdone  # node的V是否回溯完成
        self.type = Type    # 0:default, 1:Norm, 2:Dead, 3:PathFnd。用于记录是否充分探索
        self.idx = -1   # 和action绑定
        # self.rank = 0   # n叉数的第几层

class MctsState:
    def __init__(self, EnvState = None, MctsNode = None):
        self.EnvState = EnvState
        self.MctsNode = MctsNode