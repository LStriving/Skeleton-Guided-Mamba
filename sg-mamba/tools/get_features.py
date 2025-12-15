"""
@author: chenzhuokun
"""
import argparse
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import os,tarfile,io
from torchvision import utils as vutils
from PIL import Image
import skvideo.io
from torch.autograd import Variable
from collections import OrderedDict
from pytorch_i3d import InceptionI3d
os.environ['OMP_NUM_THREADS'] = "8"

#videosDir="2stages/datas/"
i3d_flow=None
i3d_rgb=None

def get_flow_frames_from_targz(tar_dir):
    list_u=[]
    list_v=[]
    with tarfile.open(tar_dir) as tar:
        mems=sorted(tar.getmembers(),key=lambda x:x.path)
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

def extractFeature(b_datas,mode):
    if mode == 'flow':
        i3d = i3d_flow
    else:
        i3d = i3d_rgb
    res=[]
    for b_data in b_datas:
        b_data = b_data.permute(0, 4, 1, 2, 3)

        if args.cuda:
            b_data = Variable(b_data.cuda(), volatile=True).float()
        else:
            b_data = Variable(b_data, volatile=True).float()

        b_features = i3d.extract_features(b_data)
        res.append(b_features.data.cpu().numpy())
    b_features=np.vstack(res)
    b_features = b_features[:,:,0,0,0]
    return b_features

def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    #input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor.permute(0,3,1,2), filename,normalize=True)

def get_features(batch_data,mode):
    full_features = [[]]
    full_features[0].append(extractFeature(batch_data,mode))
    full_features = [np.concatenate(i, axis=0) for i in full_features]
    full_features = [np.expand_dims(i, axis=0) for i in full_features]
    full_features = np.concatenate(full_features, axis=0)
    return full_features

#return rgb featrue,flow feature,frame cnt
def getVideoFeatures(videoName,args):
    rgb_datas=skvideo.io.vread(os.path.join(args.videos_dir,videoName+".avi"))
    frame_cnt=rgb_datas.shape[0]
    rgb_datas=torch.from_numpy(rgb_datas)
    rgb_datas_tmp=torch.zeros(rgb_datas.shape[0:1]+(args.img_size,args.img_size,3)).double()
    for index,rgb_data in enumerate(rgb_datas):
        rgb_data_tmp=torch.from_numpy(cv2.resize(rgb_data.numpy(),(args.img_size,args.img_size))).double()
        rgb_datas_tmp[index,:,:,:]=rgb_data_tmp
    #rgb_datas=torch.nn.functional.interpolate(rgb_datas,size=[args.img_size,args.img_size,3],mode="linear")
    rgb_datas=rgb_datas_tmp
    rgb_datas=rgb_datas.view(-1,args.img_size,args.img_size,3)
    rgb_datas=rgb_datas/127.5-1
    slidedatas_rgb=slideTensor(rgb_datas,args.chunk_size,args.frequency)
    slidedatas_rgb=get_batch_data(slidedatas_rgb,args.batch_size,3,args)
    flow_datas=get_flow_frames_from_targz(os.path.join(args.flow_dir,videoName+".tar.gz"))
    flow_datas_tmp=torch.zeros(flow_datas.shape[0:1]+(args.img_size,args.img_size,2)).double()
    for index,flow_data in enumerate(flow_datas):
        flow_data_tmp=torch.from_numpy(cv2.resize(flow_data.numpy(),(args.img_size,args.img_size))).double()
        flow_datas_tmp[index,:,:,:]=flow_data_tmp
    #flow_datas=torch.nn.functional.interpolate(flow_datas,size=[args.img_size,args.img_size,2],mode="nearest")
    flow_datas=flow_datas_tmp
    flow_datas=flow_datas.view(-1,args.img_size,args.img_size,2)
    slidedatas_flow=slideTensor(flow_datas,args.chunk_size,args.frequency)
    slidedatas_flow=get_batch_data(slidedatas_flow,args.batch_size,2,args)
    feat_spa=get_features(slidedatas_rgb,"rgb")
    feat_tem=get_features(slidedatas_flow,"flow")
    feat_spa=torch.from_numpy(feat_spa).permute(0,2,1)
    feat_tem=torch.from_numpy(feat_tem).permute(0,2,1)
    feat_spa=torch.nn.functional.interpolate(feat_spa,size=[args.feature_frame_cnt]).permute(0,2,1)
    feat_tem=torch.nn.functional.interpolate(feat_tem,size=[args.feature_frame_cnt]).permute(0,2,1)
    return [feat_spa,feat_tem,frame_cnt]

def initI3ds(args):
    global i3d_rgb,i3d_flow
    i3d_flow = InceptionI3d(400, in_channels=2)
    i3d_flow.load_state_dict(torch.load(args.flow_i3d))
    i3d_rgb = InceptionI3d(7, in_channels=3)
    new_kv=OrderedDict()
    old_kv=torch.load(args.rgb_i3d)['state_dict']
    for k,v in old_kv.items():
        new_kv[k.replace("module.","")]=v
    i3d_rgb.load_state_dict(new_kv)
    #i3d_rgb = InceptionI3d(400, in_channels=3)
    #i3d_rgb.load_state_dict(torch.load(load_model_rgb))
    i3d_rgb.train(False)
    i3d_flow.train(False)
    if args.cuda:
        i3d_rgb.cuda()
        i3d_flow.cuda()

def run(args):
    os.makedirs(args.output_dir,exist_ok=True)
    initI3ds(args)
    videoDirs=os.listdir(args.videos_dir)
    #videoDirs=[i for i in videoDirs if (i.endswith("_5.avi") or i.endswith("_6.avi"))]
    tem_outdir=os.path.join(args.output_dir,args.i3dFlowFeatureDir)
    rgb_outdir=os.path.join(args.output_dir,args.i3dFeatureDir)
    os.makedirs(tem_outdir,exist_ok=True)
    os.makedirs(rgb_outdir,exist_ok=True)
    for videoDir in tqdm(videoDirs):
        name=videoDir.split(".avi")[0]
        if os.path.exists(os.path.join(tem_outdir,name+".npz")) and os.path.exists(os.path.join(rgb_outdir,name+".npz")):
            x=1
        features=getVideoFeatures(name,args)
        feat_spa,feat_tem,frame_cnt=features
        np.savez(os.path.join(tem_outdir,name),
            feature=feat_tem,
            frame_cnt=frame_cnt,
            video_name=name,
        )
        np.savez(os.path.join(rgb_outdir,name),
            feature=feat_spa,
            frame_cnt=frame_cnt,
            video_name=name,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="./I3D/i3d_result/8_3_256_new_clip")
    parser.add_argument('--i3dFeatureDir', type=str, default="rgb_feature")
    parser.add_argument('--i3dFlowFeatureDir', type=str, default="flow_feature")
    parser.add_argument('--img_size', type=int, help='image size', default=128)
    parser.add_argument('--frequency', type=int, help='sample frequency', default=3)
    parser.add_argument('--chunk_size', type=int, help='chunk size', default=8)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--videos_dir', type=str, help='test data path', default="result/datas/")
    parser.add_argument('--flow_dir', type=str, help='test flow data path', default="result/flow_frames/")
    parser.add_argument('--cuda', action="store_true", help='if use gpu')
    parser.add_argument('--flow_i3d', type=str, help='test data path', default="I3D/models/flow_imagenet.pt")
    parser.add_argument('--rgb_i3d', type=str, help='test data path', default="/mnt/cephfs/home/nigengqin/i3d/exp/8frames_lr01_bs4_12_13/checkpoint.best.pth")
    parser.add_argument('--feature_frame_cnt', type=int, help='number of frame of a clip', default=256)
    parser.add_argument('--before_batch_cnt', type=int, help='number of frame before interplote', default=256)
    args = parser.parse_args()
    print("[Dreprecated] This script is deprecated. Please use get_features_batch.py instead.")
    run(args)