class VehicleParams:
    def __init__(self, length, width, rear_to_back, wheelbase, radius):
        self.length = length
        self.width = width
        self.rear_to_back = rear_to_back
        self.wheelbase = wheelbase
        self.radius = radius

class MapParams:
    def __init__(self, xmin, xmax, ymin, ymax, grid_num, cell_size):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.grid_num = grid_num
        self.cell_size = cell_size

class TrainingParams:
    def __init__(self, learning_rate, num_iterations, batch_size):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size

VehPara = VehicleParams(4.48, 1.85, 1.316, 2.75, 5.0)
TrainPara = TrainingParams(0.1, 10000, 500)
MapParam = MapParams(-8, 8, -4, 7, 200, 0.2)