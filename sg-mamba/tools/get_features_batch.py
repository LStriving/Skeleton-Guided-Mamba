import argparse
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import tarfile
import io
from torchvision import utils as vutils
from PIL import Image
import skvideo.io
from collections import OrderedDict
from pytorch_i3d import InceptionI3d

# Set environment variable for thread control
os.environ['OMP_NUM_THREADS'] = "8"

# Global variables for I3D models
i3d_flow = None
i3d_rgb = None

# Function to extract flow frames from a tar.gz file
def get_flow_frames_from_targz(tar_dir):
    list_u, list_v = [], []
    with tarfile.open(tar_dir) as tar:
        mems = sorted(tar.getmembers(), key=lambda x: x.path)
        for x in mems:
           if(x.size==0):
               continue
           filelikeobject=tar.extractfile(x)
           r=filelikeobject.read()
           bytes_stream = io.BytesIO(r)
           roiimg=Image.open(bytes_stream)
           nparr=np.array(roiimg,dtype=np.float)
           norm_data=nparr/127.5-1
           if(x.path.split("/")[1]=="u"):
               list_u.append(torch.tensor(norm_data))
           else:
               list_v.append(torch.tensor(norm_data))
    res_tensor=torch.stack([torch.stack(list_u),torch.stack(list_v)],dim=3)
    return res_tensor

#datas->(t,w,h,c)  output->(t_new,w,h,c)
def slideTensor(datas,window_size,step):
    start=0
    len=datas.shape[0]
    window_datas=[]
    while(start<len):
        if(start+window_size>len-1):
            break
        window_datas.append(datas[start:start+window_size,:,:,:])
        start+=step
    result=torch.stack(window_datas, 0)
    return result

def get_batch_data(slide_datas,batch_size,c,args):
    slide_datas=slide_datas.permute(2,3,0,1,4)
    slide_datas=torch.nn.functional.interpolate(slide_datas,size=[args.before_batch_cnt,8,c],mode="nearest")
    slide_datas=slide_datas.permute(2,3,0,1,4)
    slide_datas=slide_datas.view(-1,batch_size,8,args.img_size,args.img_size,c)
    return slide_datas

# Function to save tensor as an image
def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    Save a tensor as an image.
    :param input_tensor: Tensor to save
    :param filename: Output filename
    """
    assert input_tensor.shape[0] == 1 and len(input_tensor.shape) == 4
    input_tensor = input_tensor.clone().detach().to(torch.device('cpu'))
    vutils.save_image(input_tensor.permute(0, 3, 1, 2), filename, normalize=True)


def get_features(data, mode, batch_size=32):
    '''
    data: (T, 8, W, H, C)
    mode: 'rgb' or 'flow'
    return data with shape: (T, 1024)
    '''
    data = data.permute(0, 4, 1, 2, 3) # (T, C, 8, W, H)
    if batch_size == -1:
        batch_size = data.shape[0]
    batch_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    if mode == 'rgb':
        i3d = i3d_rgb
    else:
        i3d = i3d_flow
    
    all_features = []
    for batch in batch_loader:
        batch = batch.float().cuda()
        with torch.no_grad():
            features = i3d.extract_features(batch)
        # output: (batch_size, 1024, 1, 1, 1)
        features = features[:, :, 0, 0, 0].cpu().numpy()
        all_features.append(features)
    all_features = np.concatenate(all_features, axis=0)
    return all_features

def resize_data(data, image_size):
    data_tmp=torch.zeros(data.shape[0:1]+(image_size,image_size,data.shape[-1])).float()
    for index, datum in enumerate(data):
        datum_tmp = torch.from_numpy(cv2.resize(datum.numpy(),(image_size,image_size))).float()
        data_tmp[index,:,:,:] = datum_tmp
    return data_tmp

#return rgb featrue,flow feature,frame cnt
def getVideoFeatures(videoName,args):
    # 1.rgb data
    rgb_datas=skvideo.io.vread(os.path.join(args.videos_dir,videoName+".avi"))
    rgb_datas=torch.from_numpy(rgb_datas)
    ## resize the rgb_datas if needed
    if rgb_datas.shape[1]!=args.img_size or rgb_datas.shape[2]!=args.img_size:
        rgb_datas = resize_data(rgb_datas, args.img_size)
    rgb_datas=rgb_datas.view(-1,args.img_size,args.img_size,3)
    ## normalize rgb datas to [-1,1]
    rgb_datas=rgb_datas/127.5-1
    # 2.flow data
    flow_datas=get_flow_frames_from_targz(os.path.join(args.flow_dir,videoName+".tar.gz"))
    ## resize flow datas if needed
    if flow_datas.shape[1]!=args.img_size or flow_datas.shape[2]!=args.img_size:
        flow_datas = resize_data(flow_datas, args.img_size)
    flow_datas=flow_datas.view(-1,args.img_size,args.img_size,2)
    # 3. extract features
    ## saved path
    saved_path=os.path.join(args.output_dir,videoName+".npy")
    extract_features(rgb_datas, flow_datas, saved_path, 0.0, 1.0, args.chunk_size, args.frequency, branch='rgb',batch_size=args.batch_size)

def extract_features(rgb_data, 
                     flow_data, 
                     new_feat_file, 
                     start_ratio, 
                     end_ratio, 
                     win_size, 
                     win_step, 
                     preprocess=None, 
                     cropped=False, 
                     branch='rgb',
                     batch_size=-1):
    rgb_time_long = rgb_data.shape[0]
    mode = 'rgb'
    # clip
    if not cropped:
        rgb_start_idx = max(0, int(start_ratio * rgb_time_long))
        rgb_end_idx = min(rgb_time_long, int(end_ratio * rgb_time_long))
        rgb_data = rgb_data[rgb_start_idx:rgb_end_idx]
    if preprocess is not None:
        rgb_data = preprocess(rgb_data)
        mode = branch
    if branch is None or branch == '' or branch.lower() == 'none':
        # do not extract but save only
        np.save(new_feat_file, rgb_data.squeeze(-1))
        return rgb_data
    # slide
    rgb_data = slideTensor(rgb_data, win_size, win_step)
    # get feat
    feat_spa=get_features(rgb_data, mode, batch_size=batch_size)
    feat_spa=torch.from_numpy(feat_spa)

    if flow_data is not None:
        flow_time_long = flow_data.shape[0]
        if not cropped:
            flow_start_idx = max(0, int(start_ratio * flow_time_long))
            flow_end_idx = min(flow_time_long, int(end_ratio * flow_time_long))
            flow_data = flow_data[flow_start_idx:flow_end_idx]
        flow_data = slideTensor(flow_data, win_size, win_step)
        feat_tem=get_features(flow_data,"flow", batch_size=batch_size)
        feat_tem=torch.from_numpy(feat_tem)
        # concat rgb and flow features
        feat = np.concatenate([feat_spa, feat_tem], axis=1)
    else:
        feat = feat_spa
    # save feat
    np.save(new_feat_file, feat)
    return feat

# Function to initialize I3D models
def initI3ds(args):
    global i3d_rgb, i3d_flow
    i3d_flow = InceptionI3d(400, in_channels=2)
    i3d_flow.load_state_dict(torch.load(args.flow_i3d))

    i3d_rgb = InceptionI3d(7, in_channels=3)
    new_kv = OrderedDict()
    old_kv = torch.load(args.rgb_i3d)['state_dict']
    for k, v in old_kv.items():
        new_kv[k.replace("module.", "")] = v
    i3d_rgb.load_state_dict(new_kv)

    i3d_rgb.eval()
    i3d_flow.eval()

    if args.cuda:
        i3d_rgb.cuda()
        i3d_flow.cuda()

# Main function to process videos
def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    initI3ds(args)
    assert os.path.exists(args.videos_dir), f'{args.videos_dir} does not exist'
    videoDir = os.listdir(args.videos_dir)
    if args.filter_list is not None:
        with open(args.filter_list, 'r') as f:
            filter_videos = set(line.strip() for line in f)
        videoDir = [video for video in videoDir if video in filter_videos]

    for video in tqdm(videoDir):
        name=video.split(".avi")[0]
        if os.path.exists(os.path.join(args.output_dir,name+".npy")) and not args.overwrite:
            continue
        getVideoFeatures(name,args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_dir', type=str, default="result/datas/", help='Test data path')
    parser.add_argument('--flow_dir', type=str, default="result/flow_frames/", help='Test flow data path')
    parser.add_argument('--output_dir', type=str, default="./I3D/i3d_result/8_3_256_new_clip")
    parser.add_argument('--filter_list', type=str, default=None, help='Filter list for videos to process')
    parser.add_argument('--img_size', type=int, default=128, help='Image size')
    parser.add_argument('--frequency', type=int, default=3, help='Sample frequency')
    parser.add_argument('--chunk_size', type=int, default=8, help='Chunk size')
    parser.add_argument('--batch_size', type=int, default=-1, help='Batch size')
    parser.add_argument('--cuda', action="store_true", help='Use GPU if available')
    parser.add_argument('--flow_i3d', type=str, default="/mnt/cephfs/home/liyirui/project/Skeleton-Guided-Mamba/sg-mamba/ckpts/flow_imagenet.pt", help='Flow I3D model path')
    parser.add_argument('--rgb_i3d', type=str, default="/mnt/cephfs/home/liyirui/project/Skeleton-Guided-Mamba/sg-mamba/ckpts/pretrained_swallow_i3d.pth", help='RGB I3D model path')
    # parser.add_argument('--feature_frame_cnt', type=int, default=256, help='Number of frames in a clip')
    parser.add_argument('--before_batch_cnt', type=int, default=256, help='Number of frames before interpolation')
    parser.add_argument('--overwrite', action="store_true", help='Overwrite existing features')
    args = parser.parse_args()
    run(args)