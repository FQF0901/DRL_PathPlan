class VehicleParams:
    def __init__(self, length, width, rear_to_back, radius):
        self.length = length
        self.width = width
        self.rear_to_back = rear_to_back
        self.radius = radius

class TrainingParams:
    def __init__(self, learning_rate, num_iterations, batch_size):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size

VehPara = VehicleParams(4.48, 1.85, 1.316, 5.0)
TrainPara = TrainingParams(0.1, 10000, 500)