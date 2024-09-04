def process_row(row_idx, scene_data, policy_value_net, max_step, scene_pkl_file):
    scene = scene_data.iloc[row_idx]
    play_data_list = []

    DrlUtil.init_PcptGeo_info(scene)

    MT = Mcts.MctsTree()
    DrlUtil.init_mcts_info(MT)
    sim_info = MT.Simulate(max_step, policy_value_net)

    state_list, V_value_list = MT.StoreTreeInfo(sim_info)
    DrlUtil.plot_EnvMcts_info(scene_pkl_file, row_idx, MT)  # [used for debug]
    play_data_list = list(zip(state_list, V_value_list))  # Convert to list for thread safety

    return play_data_list

def process_chunk(chunk, scene_data, policy_value_net, max_step, scene_pkl_file):
    chunk_results = []
    for row_idx in chunk:
        chunk_results.extend(process_row(row_idx, scene_data, policy_value_net, max_step, scene_pkl_file))
    return chunk_results

def collection(scene_num=100, max_step=10000, deque_len=300000):
    print(utils.HighLightGreenMsg('运行 collection()'))

    policy_value_net = PolicyValueNet(model_file=os.path.join(Config.StorePath.tree_info_path, 'policy_value_net.pkl'))

    scene_file_list = DrlUtil.get_file_list_from_dir([])

    for scene_pkl_file in scene_file_list:
        with open(scene_pkl_file, 'rb') as scene_pkl_data:
            scene_data = pickle.load(scene_pkl_data)
            row_num = scene_data.shape[0]
            sampled_scene_idx_list = random.sample(range(row_num), min(scene_num, row_num))

            num_chunks = 4
            chunk_size = len(sampled_scene_idx_list) // num_chunks
            chunks = [sampled_scene_idx_list[i:i + chunk_size] for i in range(0, len(sampled_scene_idx_list), chunk_size)]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as executor:
                future_to_chunk = {executor.submit(process_chunk, chunk, scene_data, policy_value_net, max_step, scene_pkl_file): chunk for chunk in chunks}
                
                for future in concurrent.futures.as_completed(future_to_chunk):
                    try:
                        chunk_results = future.result()
                        update_data_buffer(chunk_results)
                    except Exception as e:
                        print(f"Error processing chunk: {e}")

    print('===== Mcts info generated done ! =====')
