try:
    with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
        data_file = pickle.load(data_dict)
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.data_buffer.extend(data_file['data_buffer'])
        self.iters = data_file['iters']
        del data_file
        self.iters += 1
        self.data_buffer.extend(play_data)
    print('成功载入数据')
    break
except:
    time.sleep(30)