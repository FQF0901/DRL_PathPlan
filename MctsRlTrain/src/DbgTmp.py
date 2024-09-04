from tqdm import tqdm

def process_chunk(chunk, scene_data, policy_value_net, max_step, scene_pkl_file, process_idx):
    chunk_results = []

    # tqdm 的使用不在子进程内，但可以在主进程中处理进度
    for cnt, row_idx in enumerate(chunk):
        chunk_results.extend(process_row(row_idx, scene_data, policy_value_net, max_step, scene_pkl_file))
        
        # 若需要进度条功能，可以在主进程中更新
        if cnt % 1 == 0 or cnt == len(chunk) - 1:
            print(f"Process {process_idx} Progress: {cnt}/{len(chunk)}")

    return chunk_results