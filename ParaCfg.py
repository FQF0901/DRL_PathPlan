class VehicleParams:
    def __init__(self, length, width, rear_to_back, wheelbase, radius):
        self.length = length
        self.width = width
        self.rear_to_back = rear_to_back
        self.wheelbase = wheelbase
        self.radius = radius

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

class HASParams:
    def __init__(self, xmin, xmax, ymin, ymax, grid_num, cell_size, step_size, maxEpsd):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.grid_num = grid_num
        self.cell_size = cell_size
        self.step_size = step_size
        self.maxEpsd = maxEpsd

class Node:
    def __init__(self, x, y, theta, g_cost, h_cost, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.parent = parent

class TrainingParams:
    def __init__(self, learning_rate, num_iterations, batch_size):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size

VehPara = VehicleParams(4.48, 1.85, 1.316, 2.75, 5.0)
TrainPara = TrainingParams(0.1, 10000, 500)
HASParam = HASParams(-8, 8, -4, 7, 200, 0.2, 0.4, 750)