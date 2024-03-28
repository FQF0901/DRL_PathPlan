from tqdm import tqdm
import time

# 创建一个带有后缀信息另起一行显示的进度条
with tqdm(total=100) as pbar:
    for i in range(100):
        pbar.set_description(f"Processing item {i}")
        pbar.update(1)
        pbar.set_postfix_str("status='Completed'", refresh=True)
        time.sleep(0.1)