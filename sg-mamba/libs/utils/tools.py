import torch

def slideTensor(datas,window_size,step):
    start=0
    len=datas.shape[0]
    window_datas=[]
    while(start<len):
        if(start+window_size>len-1):
            break
        window_datas.append(datas[start:start+window_size,...])
        start+=step
    result=torch.stack(window_datas, 0) # (num_windows, window_size, w, h, c)
    return result