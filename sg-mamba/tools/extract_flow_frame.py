"""
@author: chenzhuokun
"""
import logging
import os,tarfile
import time
from glob import glob
import numpy as np
import cv2
import skvideo.io
import threading
import argparse
import shutil
from tqdm import tqdm
from PIL import Image
IMG_WIDTH=128
IMG_HEIGHT=128

def cal_for_frames(video_path, img_width, img_height):
    frames=skvideo.io.vread(video_path)
    flow = []
    temp = cv2.resize(frames[0], (img_width, img_height))
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    prev=temp
    for i in range(frames.shape[0]):
        temp=cv2.resize(frames[i],(img_width,img_height))
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        curr = temp
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    res=np.array(flow)

    return res

def make_targz(source_dir,output_filename):
  with tarfile.open(output_filename, "w:") as tar:
    tar.add(source_dir, arcname=os.path.basename(source_dir))

def compute_TVL1(prev, curr, bound=20):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    # TVL1 = cv2.DualTVL1OpticalFlow_create()
    # TVL1=cv2.createOptFlow_DualTVL1()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow


def save_flow(video_flows, flow_path):
    if not os.path.exists(flow_path):
        os.mkdir(savePath)
    if not os.path.exists(os.path.join(flow_path, 'u')):
        os.mkdir(os.path.join(flow_path, 'u'))
    if not os.path.exists(os.path.join(flow_path, 'v')):
        os.mkdir(os.path.join(flow_path, 'v'))
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path, 'u', "{:06d}.png".format(i)),
                    flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path, 'v', "{:06d}.png".format(i)),
                    flow[:, :, 1])


def extract_flow(video_path, flow_path, img_width, img_height):
    flow = cal_for_frames(video_path, img_width, img_height)
    save_flow(flow, flow_path)
    make_targz(flow_path,flow_path+".tar.gz")
    shutil.rmtree(flow_path)
    return

def timeCost(timeCost):
    m, s = divmod(timeCost, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)

def parse_args():
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--video_path', type=str, help='dir path of rgb datas', default="")
    parser.add_argument('--save_path', type=str, help='dir path to save', default="")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=parse_args()
    cv2.setNumThreads(10)
    video_list=sorted(os.listdir(args.video_path))
    startTime=time.time()
    count=0
    startIndex=0
    endIndex=len(video_list)-1
    todolist=video_list
    os.makedirs(args.save_path,exist_ok=True)
    todolist=[i for i in todolist if not os.path.exists(args.save_path+"/"+i[0:len(i)-4]+".tar.gz")]
    for videoName in tqdm(todolist):
        videoPath=args.video_path+"/"+videoName
        savePath=args.save_path+"/"+videoName[0:len(videoName)-4]
        if os.path.exists(savePath):
            continue
        extract_flow(videoPath,savePath,IMG_WIDTH, IMG_HEIGHT)
        count+=1
        nowTime=time.time()