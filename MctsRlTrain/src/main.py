"""
@author: Fqf
@time: 20240807
@file: main.py
@description: main func of RL
"""

import os
import concurrent.futures
import time
from Collection import collection
from Train import training_pipeline as train

# ==========================================================
# ======================== main.py =========================
# ==========================================================

def update_input_files():
    # 实现更新输入文件的逻辑
    pass

def run_functions():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_collection = executor.submit(collection(scene_num = 100, max_step = 10000, deque_len = 100000))
        future_train = executor.submit(train.run())

        concurrent.futures.wait([future_collection, future_train])

if __name__ == "__main__":

    for _ in range(1):
        run_functions()
        update_input_files()
        time.sleep(30)

    # try:
    #     os.system('rundll32.exe powrprof.dll,SetSuspendState 0,1,0')
    #     # os.system("shutdown /s /t 0")
    # except Exception as e:
    #     print(f"An error occurred: {e}")