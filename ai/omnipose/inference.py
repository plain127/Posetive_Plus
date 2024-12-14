# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
#                                    OmniPose                                    #
#      Rochester Institute of Technology - Vision and Image Processing Lab       #
#                      Bruno Artacho (bmartacho@mail.rit.edu)                    #
# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import json
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import cython

import dataset
import models
from models.omnipose import OmniPose, get_omnipose
# from models.omnipose import get_Canny_HRNet
# from models.frankenstein import get_frankenstein
from core.inference import get_final_preds_no_transform


class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.color)):
            self.link_pairs[i].append(tuple(np.array(self.color[i])/255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i])/255.))

        # Red    = (240,  2,127)
        # Yellow = (255,255,  0)
        # Green  = (169,209,142)
        # Pink   = (252,176,243)
        # Blue   = (0,176,240)
        color_ids = [(0,176,240), (252,176,243), (169,209,142), (255,255,  0), (240,2,127)]

        self.color_ids = []
        for i in range(len(color_ids)):
            self.color_ids.append(tuple(np.array(color_ids[i])/255.))


color = [(252,176,243),(252,176,243),(252,176,243),
            (0,176,240), (0,176,240), (0,176,240),
            (240,2,127),(240,2,127),(240,2,127), (240,2,127), (240,2,127), 
            (255,255,0), (255,255,0),(169, 209, 142),
            (169, 209, 142),(169, 209, 142)]

link_pairs = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], \
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], \
    [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]]

point_color = [(240,2,127),(240,2,127),(240,2,127), 
            (240,2,127), (240,2,127), 
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (252,176,243),(0,176,240),(252,176,243),
            (0,176,240),(252,176,243),(0,176,240),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142)]

artacho_style = ColorStyle(color, link_pairs, point_color)

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize MPII predictions')
    parser.add_argument('--dataset', type=str, default='MPII')
    parser.add_argument('--image-path', help='Path of MPII val images',
                        type=str, default='/content/drive/MyDrive/mpii/images')
    parser.add_argument('--gt-anno', help='Path of MPII val annotation', type=str,
                        default='/content/drive/MyDrive/mpii/annot/valid.json')
    parser.add_argument('--save-path',help="Path to save the visualizations", type=str, default='samples/')
    parser.add_argument('--prediction', help="Prediction file to visualize", type=str, required=True)
    args = parser.parse_args()

    return args


# 수정1: 관절 json 출력을 위한 joint dict
def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)
        
    return joints_dict



def plot_MPII_image(preds, img_path, save_path, link_pairs, ring_color, color_ids, save=True):
    # Read Images
    data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    data_numpy = cv2.resize(data_numpy, (384,288), interpolation = cv2.INTER_AREA)
    h = data_numpy.shape[0]
    w = data_numpy.shape[1]
    
    # Plot
    fig = plt.figure(figsize=(w/100, h/100), dpi=100)
    ax = plt.subplot(1,1,1)
    bk = plt.imshow(data_numpy[:,:,::-1])
    bk.set_zorder(-1)
    joints_dict = map_joint_dict(preds[0])
    
    # stick 
    # for k, link_pair in enumerate(link_pairs):
    #     if link_pair[0] in joints_dict and link_pair[1] in joints_dict:
    #         lw = 2
    #         line = mlines.Line2D(
    #                 np.array([joints_dict[link_pair[0]][0],
    #                           joints_dict[link_pair[1]][0]]),
    #                 np.array([joints_dict[link_pair[0]][1],
    #                           joints_dict[link_pair[1]][1]]),
    #                 ls='-', lw=lw, alpha=1, color=color_ids[0],)
    #         line.set_zorder(0)
    #         ax.add_line(line)
            
    for k in range(preds.shape[1]):
        if preds[0,k,0] > w or preds[0,k,1] > h:
            continue
        radius = 2
        circle = mpatches.Circle(tuple(preds[0,k,:2]), 
                                 radius=radius, 
                                 ec='black', 
                                 fc=ring_color[k % len(ring_color)], 
                                 alpha=1, 
                                 linewidth=1)
        circle.set_zorder(1)
        ax.add_patch(circle)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)        
    plt.margins(0,0)
    print(save_path)
    plt.savefig(save_path, format='jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

# 수정2: json 저장 코드
def save_all_keypoints_to_json(all_preds, save_path):
    with open(save_path, 'w') as f:
        json.dump(all_preds, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name',
                        default='/model/omnipose/experiments/mpii/omnipose_w48_256x256yaml.yaml', type=str)
    parser.add_argument('--opts', help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument('--dataDir', help='data directory', type=str, default='')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')
    parser.add_argument('--model-file', help='path to the pretrained model file', required=True, type=str)
    parser.add_argument('--files-loc', help='input images directory', required=True, type=str)
    parser.add_argument('--output-dir', help='output images directory', required=True, type=str)
    parser.add_argument('--output-image-name', help='name of the output image file', required=True, type=str)
    parser.add_argument('--output-json-name', help='name of the output JSON file', required=True, type=str)

    args = parser.parse_args()
    return args

# 수정 3: 이미지 center, scale 계산을 위한 bbox 형성
def calculate_bbox_info(joints):
    visible_joints = [joint for joint in joints if joint[0] > 0 and joint[1] > 0]
    if not visible_joints:
        return (0, [0, 0])

    min_x = min(joint[0] for joint in visible_joints)
    max_x = max(joint[0] for joint in visible_joints)
    min_y = min(joint[1] for joint in visible_joints)
    max_y = max(joint[1] for joint in visible_joints)

    center = [(min_x + max_x) / 2, (min_y + max_y) / 2]
    scale = max(max_x - min_x, max_y - min_y) / 200.0

    return (scale, center)


def main(args):
    # args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.omnipose.get_omnipose')(cfg, is_train=False)

    if args.model_file:
        cfg.defrost()
        cfg.TEST.MODEL_FILE = args.model_file
        cfg.freeze()
        logger.info("=> loading checkpoint '{}'".format(cfg.TEST.MODEL_FILE))
        checkpoint = torch.load(cfg.TEST.MODEL_FILE)
        best_perf = checkpoint['perf']

        model_state_dict = model.state_dict()
        new_model_state_dict = {}
        
        for k in checkpoint['state_dict']:
            if k in model_state_dict and model_state_dict[k].size() == checkpoint['state_dict'][k].size():
                new_model_state_dict[k] = checkpoint['state_dict'][k]
            else:
                print('Skipped loading parameter {}'.format(k))

        model.load_state_dict(checkpoint, strict=False)

        print('best_perf', best_perf)

        model.load_state_dict(new_model_state_dict, strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))

        model_state_dict = torch.load(model_state_file)
        new_model_state_dict = {}
        for k in model_state_dict:
            if k in model_state_dict and model_state_dict[k].size() == model_state_dict[k].size():
                new_model_state_dict[k] = model_state_dict[k]
            else:
                print('Skipped loading parameter {}'.format(k))

        model.load_state_dict(new_model_state_dict)

    model = model.cuda()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(),normalize,])

    model.eval()

    img_path = args.files_loc
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    all_data = []

    print("Processing:", img_path)
    data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if data_numpy is None:
        print(f"Error reading image {img_path}. Skipping...")
        return

    data_numpy = cv2.resize(data_numpy, (384, 288), interpolation=cv2.INTER_AREA)
    data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

    data_numpy = transform(data_numpy)
    input = torch.zeros((1, 3, data_numpy.shape[1], data_numpy.shape[2]))
    input[0] = data_numpy

    input = input.cuda()

    outputs = model(input)
    preds, maxvals = get_final_preds_no_transform(cfg, outputs.detach().cpu().numpy())

    preds = 4 * preds

    joint_vis = [1 if maxval[0] > 0.5 else 0 for maxval in maxvals[0]]
    joints = preds[0].tolist()
    scale, center = calculate_bbox_info(joints)

    original_image = cv2.imread(img_path)
    original_size = original_image.shape[:2]  # height, width
    transformed_keypoints = transform_coords(joints, original_size)

    img_data = {
        "joint_vis": joint_vis,
        "joints": joints,
        "image": os.path.basename(img_path),
        "scale": scale,
        "center": center
    }

    all_data.append(img_data)

    colorstyle = artacho_style
    output_image_path = os.path.join(output_dir, args.output_image_name)
    plot_MPII_image(preds, img_path, output_image_path, colorstyle.link_pairs, colorstyle.ring_color, colorstyle.color_ids, save=True)

   # 수정4: pg2 입력을 위한 데이터 후처리
    output_image = cv2.imread(output_image_path)
    resized_image = cv2.resize(output_image, (64, 128), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_image_path, resized_image)

    # 수정4: 후처리된 데이터에 적합한 json 출력
    width_ratio = 64 / 384
    height_ratio = 128 / 288
    resized_joints = [[x * width_ratio, y * height_ratio] for x, y in joints]
    resized_center = [center[0] * width_ratio, center[1] * height_ratio]
    resized_scale = scale * max(width_ratio, height_ratio)  # scale은 더 큰 비율로 변환

    img_data["joints"] = resized_joints
    img_data["center"] = resized_center
    img_data["scale"] = resized_scale

    json_file_path = os.path.join(output_dir, args.output_json_name)
    with open(json_file_path, 'w') as f:
        json.dump(all_data, f, indent=4)

if __name__ == '__main__':
    args = parse_args()
    main(args)
