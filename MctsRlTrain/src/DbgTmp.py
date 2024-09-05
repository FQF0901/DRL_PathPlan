from multiprocessing import Pool, Pipe
from tqdm import tqdm
ToBatch = lambda arr,size:[arr[i*size:(i+1)*size] for i in range((size-1+len(arr))//size)]

def batch_exec_v3(f,args_batch,w=None):
    results=[]
    for i,args in enumerate(args_batch):
        ans = f(*args)
        results.append(ans)
        if w:w.send(1)
    return results

def multi_process_exec_v3(f,args_mat,pool_size=5,desc=None):
    if len(args_mat)==0:return []
    batch_size=max(1,int(len(args_mat)/4/pool_size))
    results=[]
    args_batches = ToBatch(args_mat,batch_size)
    with tqdm(total=len(args_mat), desc=desc) as pbar:
        with Pool(processes=pool_size) as pool:
            r,w=Pipe(duplex=False)
            pool_rets=[]
            for i,args_batch in enumerate(args_batches):
                pool_rets.append(pool.apply_async(batch_exec_v3,(f,args_batch,w)))
            cnt=0
            while cnt<len(args_mat):
                try:
                    msg=r.recv()
                    pbar.update(1)
                    cnt+=1
                except EOFError:
                    break
            for ret in pool_rets:
                for r in ret.get():
                    results.append(r)
    return results