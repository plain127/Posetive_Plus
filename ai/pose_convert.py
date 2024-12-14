import cv2
import json
import numpy as np
import pandas as pd

class PoseConvert:
    def __init__(self, generation_id):
        self.generation_id = generation_id
        self.ppe_path = f"/model/generation/{generation_id}/converted_target_pose.json"
        self.ppe_name = None
        self.ppe_mpii_joints = []
        self.ppe_coco_joints = []
        self.pairs = [
            (0, 14), (0, 15), (14, 16), (15, 17)
            (0, 1), (1, 2), (2, 3), (3, 4),
            (1, 5), (5, 6), (6, 7), (1, 8),
            (8, 9), (9, 10), (1, 11), (11, 12),
            (12, 13)
        ]
        
    #pe json 파일 파싱
    def load_pe(self, pe_path):
        with open(pe_path, "r") as f:
            pe_data = json.load(f)
        return pe_data
   
    def matching(self, pe_data):
        pe_joints = []
        for i in range(16):
            if pe_data[0]["joint_vis"][i] == 0:
                pe_joints.append([-1,-1])
            elif pe_data[0]["joint_vis"][i] == 1:
                pe_joints.append(pe_data[0]["joints"][i])
        
        pe_joints = [[int(value) for value in sublist] for sublist in pe_joints]
        return pe_joints

    def mpii_data(self):
        ppe_data = self.load_pe(self.ppe_path)
        self.ppe_name = "converted_" + ppe_data[0]["image"]
        self.ppe_mpii_joints = self.matching(ppe_data)

    #mpii => coco
    def convert(self, mpii_joints):
        coco_joints = []

        coco_joint_name = ['Rank', 'Rkne', 'Rhip', 'Lhip', 'Lkne', 'Lank', 'remove1',
                           'remove2' ,'neck', 'nose', 'Rwri', 'Relb', 'Rsho', 'Lsho',
                           'Lelb', 'Lwri', 'Reye', 'Leye', 'Rear', 'Lear']

        mapping1 = {coco_joint_name[i] : mpii_joints[i] for i in range(16)}
        
        mapping1['nose'][1] += 10
        reye_x = float(mapping1['nose'][0] - 2)
        reye_y = float(mapping1['nose'][1] - 3)
        
        leye_x = float(mapping1['nose'][0] + 2)
        leye_y = float(mapping1['nose'][1] - 3)

        lear_x = float(mapping1['nose'][0] + 4)
        lear_y = float(mapping1['nose'][1] - 2)

        rear_x = float(mapping1['nose'][0] - 4)
        rear_y = float(mapping1['nose'][1] - 2)

        coco_order = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 
                      'Lelb', 'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip',
                      'Lkne', 'Lank']
        
        mapping2 = {key : mapping1[key] for key in coco_order}

        for joint in list(mapping2.values()):
            coco_joints.append(joint)
        
        coco_joints.append([reye_x, reye_y])
        coco_joints.append([leye_x, leye_y])
        coco_joints.append([rear_x, rear_y])
        coco_joints.append([lear_x, lear_y])

        self.ppe_coco_joints = coco_joints

    def make_skeleton_img(self,img_size=(512,512)):
        self.mpii_data()
        self.convert(self.ppe_mpii_joints)
        
        blank_image = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8) * 255
        joints = np.array(self.ppe_coco_joints, dtype=np.float32)
        
        min_coords = joints.min(axis=0)
        max_coords = joints.max(axis=0)
        
        joints -= min_coords
        
        scale = 0.8 * min(img_size[0] / (max_coords[0] - min_coords[0], img_size[1] / (max_coords[1] - min_coords[1])))
        joints *= scale
        
        offset = ((img_size[0] - joints[:, 0].max()) / 2, (img_size[1] - joints[: ,1].max()) / 2)
        joints[:, 0] += offset[0]
        joints[:, 1] += offset[1]
        
        for _, (x, y) in enumerate(joints):
            cv2.circle(blank_image, (int(x), int(y)), 5, (0, 0, 0), -1)
        
        for start, end in self.pairs:
            start_point = tuple(joints[start].astype(int))
            end_point = tuple(joints[end].astype(int))
            cv2.line(blank_image, start_point, end_point, (0, 0, 255), 2)
            
        cv2.imwrite(f'model/generation/{self.generation_id}ppe_img.jpg', blank_image)
        