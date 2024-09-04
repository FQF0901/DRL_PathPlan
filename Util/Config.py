"""
@author: Fqf
@time: 20240724
@file: Config.py
@description: shared parameters
"""

# ==================================================================
# ==================== Global Config Parameters ====================
# ==================================================================

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

class MultiTreadNum:
    def __init__(self) -> None:
        self.collection_multi_thread_num = 4
        self.Convert2DataSet_multi_thread_num = 4

class StoreFolderPath:
    def __init__(self) -> None:
        self.mf4_raw_file_path =        r'E:\DataSet\Mf4RawFile'
        self.scene_slice_data_path =    r'E:\DataSet\SceneSliceData'
        self.tree_info_path =           r'E:\DataSet\TreeinfoData'
        self.train_dataset_path =       r'E:\DataSet\TrainDataSet'
        self.net_path =                 r'E:\DataSet\TrainDataSet'


VehPara = VehParams()
StorePath = StoreFolderPath()
MultiTread = MultiTreadNum()
