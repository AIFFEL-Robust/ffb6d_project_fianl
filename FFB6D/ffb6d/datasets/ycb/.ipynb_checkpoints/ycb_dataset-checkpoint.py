#!/usr/bin/env python3
import os
import cv2
import torch
import os.path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from common import Config
import pickle as pkl
from utils.basic_utils import Basic_Utils
import scipy.io as scio
import scipy.misc
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except:
    from cv2 import imshow, waitKey
import normalSpeed
from models.RandLA.helper_tool import DataProcessing as DP

#for ycb_m
import json
import math

# ycb-m segment value transform for ycb-v
#seg_id={'002': 12, '003': 24, '004': 36, '005': 48,'006':60,'007':72,'008':84,'009':96,
#        '010':108, '011':120, '019':132, '021':144, '024':156, '025':168, '035':180,
#        '036':192, '037':204, '040':216, '051':228, '052':240,'061':252}

# ycb-m segment value transform for ycb-v
seg_id={1: 12, 2: 24, 3: 36, 4: 48, 5:60, 6:72, 7:84, 8:96,
        9:108, 10:120, 11:132, 12:144, 13:156, 14:168, 15:180,
        16:192, 17:204, 18:216, 19:228, 20:240, 21:252}

# ycb-m class id
class_id_list = {2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 19:11, 21:12,
                 24:13, 25:14, 35:15, 36:16, 37:17, 40:18, 51:19, 52:20, 61:21}

# fat fixed_model_transform (ycbm to ycb)
cls_fixed_transform= {1: [[ 86.602500915527344, 0, 50, 0 ],
                          [ -50, 0, 86.602500915527344, 0 ],
                          [ 0, -100, 0, 0 ],
                          [ 0.99080002307891846, 6.9902000427246094, 1.6902999877929688, 1 ]],
                      2:[[ 0, 0, 100, 0 ],
                         [ -100, 0, 0, 0 ],
                         [ 0, -100, 0, 0 ],
                         [ -1.4141999483108521, 10.347499847412109, 1.2884999513626099, 1 ]],
                      3:[[ -3.4877998828887939, 3.4899001121520996, 99.878196716308594, 0 ],
                         [ -99.926002502441406, -1.7441999912261963, -3.4284999370574951, 0 ],
                         [ 1.6224000453948975, -99.923896789550781, 3.5481998920440674, 0 ],
                         [ -1.795199990272522, 8.7579002380371094, 0.38839998841285706, 1 ]],
                      4:[[ 99.144500732421875, 0, -13.052599906921387, 0 ],
                         [ 13.052599906921387, 0, 99.144500732421875, 0 ],
                         [ 0, -100, 0, 0 ],
                         [ -0.1793999969959259, 5.1006999015808105, -8.4443998336791992, 1 ]],
                      5: [[ 92.050498962402344, 0, 39.073101043701172, 0 ],
                          [ -39.073101043701172, 0, 92.050498962402344, 0 ],
                          [ 0, -100, 0, 0 ],
                          [ 0.49259999394416809, 9.2497997283935547, 2.7135999202728271, 1 ]],
                      6:[[ 100, 0, 0, 0 ],
                         [ 0, 0, 100, 0 ],
                         [ 0, -100, 0, 0 ],
                         [ 2.6048998832702637, 1.3551000356674194, 2.2132000923156738, 1 ]],
                      7: [[ -88.264503479003906, 46.947200775146484, -2.3113000392913818, 0 ],
                          [ -46.878200531005859, -88.281303405761719, -2.9734001159667969, 0 ],
                          [ -3.4363000392913818, -1.5410000085830688, 99.929100036621094, 0 ],
                          [ 1.010200023651123, 1.6993999481201172, -1.7572000026702881, 1 ]],
                      8: [[ 22.494199752807617, 97.436996459960938, 0.19629999995231628, 0 ],
                          [ -97.433296203613281, 22.495100021362305, -0.85030001401901245, 0 ],
                          [ -0.87269997596740723, 0, 99.996200561523438, 0 ],
                          [ -0.29069998860359192, 2.3998000621795654, -1.4543999433517456, 1 ]],
                      9: [[ 99.862998962402344, 0, -5.2336001396179199, 0 ],
                          [ 5.2336001396179199, 0, 99.862998962402344, 0 ],
                          [ 0, -100, 0, 0 ],
                          [ 3.4065999984741211, 3.8584001064300537, 2.4767000675201416, 1 ]],
                      10: [[ -36.395401000976563, -17.364799499511719, 91.508697509765625, 0 ],
                           [ -93.087699890136719, 3.4368999004364014, -36.371200561523438, 0 ],
                           [ 3.1707000732421875, -98.420799255371094, -17.415399551391602, 0 ],
                           [ 0.036600001156330109, 1.497499942779541, 0.44449999928474426, 1 ]],
                      11: [[ 71.933998107910156, 0, -69.465797424316406, 0 ],
                           [ 69.465797424316406, 0, 71.933998107910156, 0 ],
                           [ 0, -100, 0, 0 ],
                           [ -2.4330000877380371, 11.837100028991699, -3.410599946975708, 1 ]],
                      12: [[ -100, 0, 0, 0 ],
                           [ 0, 0, -100, 0 ],
                           [ 0, -100, 0, 0 ],
                           [ -2.1663999557495117, 12.483799934387207, 1.1708999872207642, 1 ]],
                      13:[[ 0, 0, 100, 0 ],
                          [ -100, 0, 0, 0 ],
                          [ 0, -100, 0, 0 ],
                          [ -4.3688998222351074, 2.6974000930786133, 1.4838999509811401, 1 ]],
                      14: [[ 99.862899780273438, 0, 5.2336001396179199, 0 ],
                           [ -5.2336001396179199, 0, 99.862899780273438, 0 ],
                           [ 0, -100, 0, 0 ],
                           [ 0.97670000791549683, 4.0118999481201172, -1.6074999570846558, 1 ]],
                      15: [[ 99.923896789550781, 1.7452000379562378, -3.4893999099731445, 0 ],
                           [ 1.6521999835968018, -99.95050048828125, -2.6770000457763672, 0 ],
                           [ -3.5343999862670898, 2.6173000335693359, -99.9031982421875, 0 ],
                           [ 4.5402998924255371, 1.1200000047683716, 2.2990999221801758, 1 ]],
                      16: [[ -22.494199752807617, 0.87269997596740723, 97.433296203613281, 0 ],
                           [ -97.436996459960938, 0, -22.495100021362305, 0 ],
                           [ -0.19629999995231628, -99.996200561523438, 0.85030001401901245, 0 ],
                           [ -0.52619999647140503, 10.234100341796875, -2.6282000541687012, 1 ]],
                      17:[[ -27.525999069213867, 96.126197814941406, -1.4426000118255615, 0 ],
                          [ -96.136802673339844, -27.525999069213867, 0.20250000059604645, 0 ],
                          [ -0.20250000059604645, 1.4426000118255615, 99.989402770996094, 0 ],
                          [ 4.5185999870300293, -1.2343000173568726, -0.71310001611709595, 1 ]],
                      18:[[ -6.1027998924255371, 2.6177000999450684, 99.779296875, 0 ],
                          [ -0.15979999303817749, -99.9656982421875, 2.6127998828887939, 0 ],
                          [ 99.813499450683594, 0, 6.1048998832702637, 0 ],
                          [ -1.1348999738693237, -0.40759998559951782, 3.5237998962402344, 1 ]],
                      19:[[ -14.553999900817871, -98.32550048828125, 10.96720027923584, 0 ],
                          [ 98.754302978515625, -15.107999801635742, -4.3982000350952148, 0 ],
                          [ 5.9814000129699707, 10.190500259399414, 99.299400329589844, 0 ],
                          [ 1.027400016784668, -0.8101000189781189, -1.802299976348877, 1 ]],
                      20:[[ 99.357200622558594, -11.320300102233887, 0, 0 ],
                          [ 11.318599700927734, 99.342002868652344, -1.7452000379562378, 0 ],
                          [ 0.19760000705718994, 1.7339999675750732, 99.98480224609375, 0 ],
                          [ 2.7757999897003174, 3.1777999401092529, -1.8323999643325806, 1 ]],
                      21:[[ 0, 0, 100, 0 ],
                          [ -100, 0, 0, 0 ],
                          [ 0, -100, 0, 0 ],
                          [ 1.7200000286102295, 2.5250999927520752, 1.8260999917984009, 1 ]]}



config = Config(ds_name='ycb')
bs_utils = Basic_Utils(config)

# ycb_m을 위한 json data 가공용 function
def from_json_cls_list(meta):
    cl_lst = []
    for i in range(len(meta['objects'])):
        cls_idx = int(meta['objects'][i]['class'][:3])
        cl_lst.append(class_id_list[cls_idx])

    cl_lst = np.array(cl_lst)
    cl_lst = cl_lst.flatten().astype(np.uint32)
    
    return cl_lst


# ycb_m and fat
def pose_transform_permuted_from_json(meta,i, cls_id, fixed=cls_fixed_transform):
    # type: (dict) -> (np.array, np.array)
    """
    Parses object pose from "pose_transform_permuted". Equivalent to parse_object_pose.

    *Note:*  Like the `fixed_model_transform`, the `pose_transform_permuted` is actually the transpose of the matrix.
    Moreover, after transposing, the columns are permuted, and there is a sign flip (due to UE4's use of a lefthand
    coordinate system).  Specifically, if `A` is the matrix given by `pose_transform_permuted`, then actual transform
    is given by `A^T * P`, where `^T` denotes transpose, `*` denotes matrix multiplication, and the permutation matrix
    `P` is given by

        [ 0  0  1]
    P = [ 1  0  0]
        [ 0 -1  0]

    :param scene_object_json: JSON fragment of a single scene object
    :return (translation, rotation) in meters
    """
    
    pose_transform = np.transpose(np.array(meta['objects'][i]['pose_transform_permuted']))
    to_righthand = np.array([[0, 0, 1],
                             [1, 0, 0],
                             [0, -1, 0]])
    r = pose_transform[:3, :3].dot(to_righthand)
    t = pose_transform[:3, 3].reshape(3,1)
    
    fixed=np.array(fixed[cls_id])
    
    fixed = fixed.T
    r_fixed = fixed[:3,:3]
    r = np.matmul(r,r_fixed)
    
    return r, t



class Dataset():

    def __init__(self, dataset_name, DEBUG=False):
        self.dataset_name = dataset_name
        self.debug = DEBUG
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        self.diameters = {}
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224])
        self.cls_lst = bs_utils.read_lines(config.ycb_cls_lst_p)
        self.obj_dict = {}
        for cls_id, cls in enumerate(self.cls_lst, start=1):
            self.obj_dict[cls] = cls_id
        self.rng = np.random
        if dataset_name == 'train':
            self.add_noise = True
            self.path = 'datasets/ycb/dataset_config/train_data_list.txt'
            self.all_lst = bs_utils.read_lines(self.path)
            self.minibatch_per_epoch = len(self.all_lst) // config.mini_batch_size
            self.real_lst = []
            self.syn_lst = []
            self.fat_lst = []
            
            # only ycbm train, only fat train
            for item in self.all_lst:
                self.real_lst.append(item)
            
            #  ycbm+fat+syn
            # for item in self.all_lst:
            #     # debuging
            #     if item[:3] == 'fat':
            #         self.fat_lst.append(item)
            #     elif item[:8] == 'data_syn':
            #         self.syn_lst.append(item)
            #         #print(item[:8], 'syn_data')
            #     else:
            #         self.real_lst.append(item)
            #         #print(item[:8], 'real_data')
                    
        else:
            self.pp_data = None
            self.add_noise = False
            self.path = 'datasets/ycb/dataset_config/test_data_list.txt'
            self.all_lst = bs_utils.read_lines(self.path)
        print("{}_dataset_size: ".format(dataset_name), len(self.all_lst))
        self.root = config.ycb_root
        self.sym_cls_ids = [13, 16, 19, 20, 21]
        
    # only ycbm train, only fat train
    def real_syn_gen(self):
        n = len(self.real_lst)
        idx = self.rng.randint(0, n)
        item = self.real_lst[idx]
        
        return item
    
# ycbm + fat + syn combine train
#     def real_syn_gen(self):
#         if self.rng.rand() > 0.9:
#             n = len(self.real_lst)
#             idx = self.rng.randint(0, n)
#             item = self.real_lst[idx]
        
#         elif self.rng.rand() > 0.7 and self.rng.rand() <= 0.9 :
#             n = len(self.fat_lst)
#             idx = self.rng.randint(0, n)
#             item = self.fat_lst[idx]
            
#         else:
#             n = len(self.syn_lst)
#             idx = self.rng.randint(0, n)
#             item = self.syn_lst[idx]
#         return item

    def real_gen(self):
        n = len(self.real_lst)
        idx = self.rng.randint(0, n)
        item = self.real_lst[idx]
        return item

    def rand_range(self, rng, lo, hi):
        return rng.rand()*(hi-lo)+lo

    def gaussian_noise(self, rng, img, sigma):
        """add gaussian noise of given sigma to image"""
        img = img + rng.randn(*img.shape) * sigma
        img = np.clip(img, 0, 255).astype('uint8')
        return img

    def linear_motion_blur(self, img, angle, length):
        """:param angle: in degree"""
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        a = int(max(list(map(abs, (dx, dy)))) * length * 2)
        if a <= 0:
            return img
        kern = np.zeros((a, a))
        cx, cy = a // 2, a // 2
        dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
        cv2.line(kern, (cx, cy), (dx, dy), 1.0)
        s = kern.sum()
        if s == 0:
            kern[cx, cy] = 1.0
        else:
            kern /= s
        return cv2.filter2D(img, -1, kern)

    def rgb_add_noise(self, img):
        rng = self.rng
        # apply HSV augmentor
        if rng.rand() > 0:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.uint16)
            hsv_img[:, :, 1] = hsv_img[:, :, 1] * self.rand_range(rng, 1.25, 1.45)
            hsv_img[:, :, 2] = hsv_img[:, :, 2] * self.rand_range(rng, 1.15, 1.35)
            hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)
            img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if rng.rand() > .8:  # sharpen
            kernel = -np.ones((3, 3))
            kernel[1, 1] = rng.rand() * 3 + 9
            kernel /= kernel.sum()
            img = cv2.filter2D(img, -1, kernel)

        if rng.rand() > 0.8:  # motion blur
            r_angle = int(rng.rand() * 360)
            r_len = int(rng.rand() * 15) + 1
            img = self.linear_motion_blur(img, r_angle, r_len)

        if rng.rand() > 0.8:
            if rng.rand() > 0.2:
                img = cv2.GaussianBlur(img, (3, 3), rng.rand())
            else:
                img = cv2.GaussianBlur(img, (5, 5), rng.rand())

        if rng.rand() > 0.2:
            img = self.gaussian_noise(rng, img, rng.randint(15))
        else:
            img = self.gaussian_noise(rng, img, rng.randint(25))

        if rng.rand() > 0.8:
            img = img + np.random.normal(loc=0.0, scale=7.0, size=img.shape)

        return np.clip(img, 0, 255).astype(np.uint8)

    def add_real_back(self, rgb, labels, dpt, dpt_msk):
        real_item = self.real_gen()
        with Image.open(os.path.join(self.root, real_item+'.depth.png')) as di:
            real_dpt = np.array(di)
        with Image.open(os.path.join(self.root, real_item+'.seg.png')) as li:
            bk_label = np.array(li)
        for key,value in seg_id.items():
            if value in labels:
                labels[labels/value==1]=int(key)
        bk_label = (bk_label <= 0).astype(rgb.dtype)
        bk_label_3c = np.repeat(bk_label[:, :, None], 3, 2)
        with Image.open(os.path.join(self.root, real_item+'.jpg')) as ri:
            back = np.array(ri)[:, :, :3] * bk_label_3c
        dpt_back = real_dpt.astype(np.float32) * bk_label.astype(np.float32)

        msk_back = (labels <= 0).astype(rgb.dtype)
        msk_back = np.repeat(msk_back[:, :, None], 3, 2)
        rgb = rgb * (msk_back == 0).astype(rgb.dtype) + back * msk_back

        dpt = dpt * (dpt_msk > 0).astype(dpt.dtype) + \
            dpt_back * (dpt_msk <= 0).astype(dpt.dtype)
        return rgb, dpt
    
    def center_crop(self, img, w, h):
        img_width, img_height = img.size
        left, right = (img_width - w) / 2, (img_width + w) / 2
        top, bottom = (img_height - h) / 2, (img_height + h) / 2
        left, top = round(max(0, left)), round(max(0, top))
        right, bottom = round(min(img_width - 0, right)), round(min(img_height - 0, bottom))
        
        return img.crop((left, top, right, bottom))

    def dpt_2_pcld(self, dpt, cam_scale, K):
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        dpt = dpt.astype(np.float32) / cam_scale
        msk = (dpt > 1e-8).astype(np.float32)
        row = (self.ymap - K[0][2]) * dpt / K[0][0]
        col = (self.xmap - K[1][2]) * dpt / K[1][1]
        dpt_3d = np.concatenate(
            (row[..., None], col[..., None], dpt[..., None]), axis=2
        )
        dpt_3d = dpt_3d * msk[:, :, None]
        return dpt_3d

    def get_item(self, item_name):
        # 각 데이터 특성에 따라 분기로 data load
        if item_name[:3] == 'dat':
            with Image.open(os.path.join(self.root, item_name+'-depth.png')) as di:
                dpt_um = np.array(di)
            with Image.open(os.path.join(self.root, item_name+'-label.png')) as li:
                labels = np.array(li)
            with Image.open(os.path.join(self.root, item_name+'-color.png')) as ri:
                if self.add_noise:
                    ri = self.trancolor(ri)
                rgb = np.array(ri)[:, :, :3]
            meta = scio.loadmat(os.path.join(self.root, item_name+'-meta.mat'))
            cls_id_lst = meta['cls_indexes'].flatten().astype(np.uint32)
            
        elif item_name[:3] == 'fat':
            #print('fat', item_name)
            with Image.open(os.path.join(self.root, item_name+'.depth.png')) as di:
                di = self.center_crop(di, 640, 480)
                dpt_um = np.array(di)
            with Image.open(os.path.join(self.root, item_name+'.seg.png')) as li:
                li = self.center_crop(li, 640, 480)
                labels = np.array(li)
            for key,value in seg_id.items():
                if value in labels:
                    labels[labels/value==1]=int(key)
            with Image.open(os.path.join(self.root, item_name+'.jpg')) as ri:
                ri = self.center_crop(ri, 640, 480)
                if self.add_noise:
                    ri = self.trancolor(ri)
                rgb = np.array(ri)[:, :, :3]
            # json
            j_path = os.path.join(self.root, item_name+'.json')
            with open(j_path) as j_data_file:
                meta=json.load(j_data_file)
            cls_id_lst = from_json_cls_list(meta) 
            
        else:
            #print('ycbm', item_name)
            with Image.open(os.path.join(self.root, item_name+'.depth.png')) as di:
                dpt_um = np.array(di)
            with Image.open(os.path.join(self.root, item_name+'.seg.png')) as li:
                labels = np.array(li)
            for key,value in seg_id.items():
                if value in labels:
                    labels[labels/value==1]=int(key)
            with Image.open(os.path.join(self.root, item_name+'.jpg')) as ri:
                if self.add_noise:
                    ri = self.trancolor(ri)
                rgb = np.array(ri)[:, :, :3]
            # json
            j_path = os.path.join(self.root, item_name+'.json')
            with open(j_path) as j_data_file:
                meta=json.load(j_data_file)
            cls_id_lst = from_json_cls_list(meta) 
        
        rgb_labels = labels.copy()
        
        #debuging
        #print('카메라', item_name[6])
        
        if item_name[6] == 'a':
            K = config.intrinsic_matrix['astra']
        elif item_name[6] == 'r':
            K = config.intrinsic_matrix['realsense']
        elif item_name[6] == 'x':
            K = config.intrinsic_matrix['xtion']
        elif item_name[:3] == 'fat':
            K = config.intrinsic_matrix['fat']
        else:
            K = config.intrinsic_matrix['ycb_K1']
        
        rnd_typ = 'syn' if 'syn' in item_name else 'real'
        
        # YCB-V data load 용
        #cam_scale = meta['factor_depth'].astype(np.float32)[0][0]
        cam_scale = 10000.0
        msk_dp = dpt_um > 1e-6

        if self.add_noise and rnd_typ == 'syn':
            rgb = self.rgb_add_noise(rgb)
            rgb, dpt_um = self.add_real_back(rgb, rgb_labels, dpt_um, msk_dp)
            if self.rng.rand() > 0.8:
                rgb = self.rgb_add_noise(rgb)
                
            # 백그라운 합성 이미지 확인용
            # sample_image_check (hjw)
#             if self.rng.rand() > 0.9999:
#                 sample_syn_path = os.getenv("HOME")+'/workspace/fmodel/FFB6D/ffb6d/train_log/ycb/syn_image_sample'
#                 new_item_name = item_name.replace("/","_")
              
#                 rgb_img = Image.fromarray(rgb)
#                 dpt_img = Image.fromarray(dpt_um)
                   
#                 rgb_img.save(os.path.join(sample_syn_path, new_item_name+'_rgb_sample.tiff'), 'tiff')
#                 dpt_img.save(os.path.join(sample_syn_path, new_item_name+'_dpt_sample.tiff'), 'tiff')

        dpt_um = bs_utils.fill_missing(dpt_um, cam_scale, 1)
        msk_dp = dpt_um > 1e-6

        dpt_mm = (dpt_um.copy()/10).astype(np.uint16)
        nrm_map = normalSpeed.depth_normal(
            dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False
        )
        if self.debug:
            show_nrm_map = ((nrm_map + 1.0) * 127).astype(np.uint8)
            imshow("nrm_map", show_nrm_map)

        dpt_m = dpt_um.astype(np.float32) / cam_scale
        dpt_xyz = self.dpt_2_pcld(dpt_m, 1.0, K)

        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
        if len(choose) < 400:
            return None
        choose_2 = np.array([i for i in range(len(choose))])
        if len(choose_2) < 400:
            return None
        if len(choose_2) > config.n_sample_points:
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[:config.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]
        else:
            choose_2 = np.pad(choose_2, (0, config.n_sample_points-len(choose_2)), 'wrap')
        choose = np.array(choose)[choose_2]

        sf_idx = np.arange(choose.shape[0])
        np.random.shuffle(sf_idx)
        choose = choose[sf_idx]

        cld = dpt_xyz.reshape(-1, 3)[choose, :]
        rgb_pt = rgb.reshape(-1, 3)[choose, :].astype(np.float32)
        nrm_pt = nrm_map[:, :, :3].reshape(-1, 3)[choose, :]
        labels_pt = labels.flatten()[choose]
        choose = np.array([choose])
        cld_rgb_nrm = np.concatenate((cld, rgb_pt, nrm_pt), axis=1).transpose(1, 0)


        if 'data' in item_name:
            RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst = self.get_pose_gt_info(cld, labels_pt, cls_id_lst, meta)
        else:
            RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst = self.get_pose_gt_info_ycbm(cld, labels_pt, cls_id_lst, meta)

        h, w = rgb_labels.shape
        dpt_6c = np.concatenate((dpt_xyz, nrm_map[:, :, :3]), axis=2).transpose(2, 0, 1)
        rgb = np.transpose(rgb, (2, 0, 1)) # hwc2chw

        xyz_lst = [dpt_xyz.transpose(2, 0, 1)] # c, h, w
        msk_lst = [dpt_xyz[2, :, :] > 1e-8]

        for i in range(3):
            scale = pow(2, i+1)
            nh, nw = h // pow(2, i+1), w // pow(2, i+1)
            ys, xs = np.mgrid[:nh, :nw]
            xyz_lst.append(xyz_lst[0][:, ys*scale, xs*scale])
            msk_lst.append(xyz_lst[-1][2, :, :] > 1e-8)
        sr2dptxyz = {
            pow(2, ii): item.reshape(3, -1).transpose(1, 0) for ii, item in enumerate(xyz_lst)
        }
        sr2msk = {
            pow(2, ii): item.reshape(-1) for ii, item in enumerate(msk_lst)
        }

        rgb_ds_sr = [4, 8, 8, 8]
        n_ds_layers = 4
        pcld_sub_s_r = [4, 4, 4, 4]
        inputs = {}
        # DownSample stage
        for i in range(n_ds_layers):
            nei_idx = DP.knn_search(
                cld[None, ...], cld[None, ...], 16
            ).astype(np.int32).squeeze(0)
            sub_pts = cld[:cld.shape[0] // pcld_sub_s_r[i], :]
            pool_i = nei_idx[:cld.shape[0] // pcld_sub_s_r[i], :]
            up_i = DP.knn_search(
                sub_pts[None, ...], cld[None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['cld_xyz%d'%i] = cld.astype(np.float32).copy()
            inputs['cld_nei_idx%d'%i] = nei_idx.astype(np.int32).copy()
            inputs['cld_sub_idx%d'%i] = pool_i.astype(np.int32).copy()
            inputs['cld_interp_idx%d'%i] = up_i.astype(np.int32).copy()
            nei_r2p = DP.knn_search(
                sr2dptxyz[rgb_ds_sr[i]][None, ...], sub_pts[None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_ds_nei_idx%d'%i] = nei_r2p.copy()
            nei_p2r = DP.knn_search(
                sub_pts[None, ...], sr2dptxyz[rgb_ds_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_ds_nei_idx%d'%i] = nei_p2r.copy()
            cld = sub_pts

        n_up_layers = 3
        rgb_up_sr = [4, 2, 2]
        for i in range(n_up_layers):
            r2p_nei = DP.knn_search(
                sr2dptxyz[rgb_up_sr[i]][None, ...],
                inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_up_nei_idx%d'%i] = r2p_nei.copy()
            p2r_nei = DP.knn_search(
                inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...],
                sr2dptxyz[rgb_up_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_up_nei_idx%d'%i] = p2r_nei.copy()

        show_rgb = rgb.transpose(1, 2, 0).copy()[:, :, ::-1]
        if self.debug:
            for ip, xyz in enumerate(xyz_lst):
                pcld = xyz.reshape(3, -1).transpose(1, 0)
                p2ds = bs_utils.project_p3d(pcld, cam_scale, K)
                print(show_rgb.shape, pcld.shape)
                srgb = bs_utils.paste_p2ds(show_rgb.copy(), p2ds, (0, 0, 255))
                imshow("rz_pcld_%d" % ip, srgb)
                p2ds = bs_utils.project_p3d(inputs['cld_xyz%d'%ip], cam_scale, K)
                srgb1 = bs_utils.paste_p2ds(show_rgb.copy(), p2ds, (0, 0, 255))
                imshow("rz_pcld_%d_rnd" % ip, srgb1)

        item_dict = dict(
            rgb=rgb.astype(np.uint8),  # [c, h, w]
            cld_rgb_nrm=cld_rgb_nrm.astype(np.float32),  # [9, npts]
            choose=choose.astype(np.int32),  # [1, npts]
            labels=labels_pt.astype(np.int32),  # [npts]
            rgb_labels=rgb_labels.astype(np.int32),  # [h, w]
            dpt_map_m=dpt_m.astype(np.float32),  # [h, w]
            RTs=RTs.astype(np.float32),
            kp_targ_ofst=kp_targ_ofst.astype(np.float32),
            ctr_targ_ofst=ctr_targ_ofst.astype(np.float32),
            cls_ids=cls_ids.astype(np.int32),
            ctr_3ds=ctr3ds.astype(np.float32),
            kp_3ds=kp3ds.astype(np.float32),
            # temp
            K = K.astype(np.float32),
        )
        item_dict.update(inputs)
        if self.debug:
            extra_d = dict(
                dpt_xyz_nrm=dpt_6c.astype(np.float32),  # [6, h, w]
                cam_scale=np.array([cam_scale]).astype(np.float32),
                K=K.astype(np.float32),
            )
            item_dict.update(extra_d)
            item_dict['normal_map'] = nrm_map[:, :, :3].astype(np.float32)
        return item_dict

    # ycbm 및 fat data label load
    def get_pose_gt_info_ycbm(self, cld, labels, cls_id_lst, meta):
        RTs = np.zeros((config.n_objects, 3, 4))
        kp3ds = np.zeros((config.n_objects, config.n_keypoints, 3))
        ctr3ds = np.zeros((config.n_objects, 3))
        cls_ids = np.zeros((config.n_objects, 1))
        kp_targ_ofst = np.zeros((config.n_sample_points, config.n_keypoints, 3))
        ctr_targ_ofst = np.zeros((config.n_sample_points, 3))
        for i, cls_id in enumerate(cls_id_lst):
            
            # original code
            #r = meta['poses'][:, :, i][:, 0:3]
            #t = np.array(meta['poses'][:, :, i][:, 3:4].flatten()[:, None])
            r, t= pose_transform_permuted_from_json(meta, i, cls_id)
            
            #debuging
            #print('r', r)
            #print('t', t)
            
            r = r/100
            t = t/100
            
            RT = np.concatenate((r, t), axis=1)
            RTs[i] = RT

            ctr = bs_utils.get_ctr(self.cls_lst[cls_id-1]).copy()[:, None]
            ctr = np.dot(ctr.T, r.T) + t[:, 0]
            ctr3ds[i, :] = ctr[0]
            msk_idx = np.where(labels == cls_id)[0]

            target_offset = np.array(np.add(cld, -1.0*ctr3ds[i, :]))
            ctr_targ_ofst[msk_idx,:] = target_offset[msk_idx, :]
            cls_ids[i, :] = np.array([cls_id])

            key_kpts = ''
            if config.n_keypoints == 8:
                kp_type = 'farthest'
            else:
                kp_type = 'farthest{}'.format(config.n_keypoints)
            kps = bs_utils.get_kps(
                self.cls_lst[cls_id-1], kp_type=kp_type, ds_type='ycb'
            ).copy()
            kps = np.dot(kps, r.T) + t[:, 0]
            kp3ds[i] = kps

            target = []
            for kp in kps:
                target.append(np.add(cld, -1.0*kp))
            target_offset = np.array(target).transpose(1, 0, 2)  # [npts, nkps, c]
            kp_targ_ofst[msk_idx, :, :] = target_offset[msk_idx, :, :]
        return RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst

    def get_pose_gt_info(self, cld, labels, cls_id_lst, meta):
        RTs = np.zeros((config.n_objects, 3, 4))
        kp3ds = np.zeros((config.n_objects, config.n_keypoints, 3))
        ctr3ds = np.zeros((config.n_objects, 3))
        cls_ids = np.zeros((config.n_objects, 1))
        kp_targ_ofst = np.zeros((config.n_sample_points, config.n_keypoints, 3))
        ctr_targ_ofst = np.zeros((config.n_sample_points, 3))
        for i, cls_id in enumerate(cls_id_lst):
            r = meta['poses'][:, :, i][:, 0:3]
            t = np.array(meta['poses'][:, :, i][:, 3:4].flatten()[:, None])
            RT = np.concatenate((r, t), axis=1)
            RTs[i] = RT

            ctr = bs_utils.get_ctr(self.cls_lst[cls_id-1]).copy()[:, None]
            ctr = np.dot(ctr.T, r.T) + t[:, 0]
            ctr3ds[i, :] = ctr[0]
            msk_idx = np.where(labels == cls_id)[0]

            target_offset = np.array(np.add(cld, -1.0*ctr3ds[i, :]))
            ctr_targ_ofst[msk_idx,:] = target_offset[msk_idx, :]
            cls_ids[i, :] = np.array([cls_id])

            key_kpts = ''
            if config.n_keypoints == 8:
                kp_type = 'farthest'
            else:
                kp_type = 'farthest{}'.format(config.n_keypoints)
            kps = bs_utils.get_kps(
                self.cls_lst[cls_id-1], kp_type=kp_type, ds_type='ycb'
            ).copy()
            kps = np.dot(kps, r.T) + t[:, 0]
            kp3ds[i] = kps

            target = []
            for kp in kps:
                target.append(np.add(cld, -1.0*kp))
            target_offset = np.array(target).transpose(1, 0, 2)  # [npts, nkps, c]
            kp_targ_ofst[msk_idx, :, :] = target_offset[msk_idx, :, :]
        return RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst
    
    
    def __len__(self):
        return len(self.all_lst)

    def __getitem__(self, idx):
        if self.dataset_name == 'train':
            item_name = self.real_syn_gen()
            data = self.get_item(item_name)
            while data is None:
                item_name = self.real_syn_gen()
                data = self.get_item(item_name)
            return data
        else:
            item_name = self.all_lst[idx]
            return self.get_item(item_name)


def main():
    # config.mini_batch_size = 1
    global DEBUG
    DEBUG = True
    ds = {}
    ds['train'] = Dataset('train', DEBUG=True)
    # ds['val'] = Dataset('validation')
    ds['test'] = Dataset('test', DEBUG=True)
    idx = dict(
        train=0,
        val=0,
        test=0
    )
    while True:
        # for cat in ['val', 'test']:
        # for cat in ['train']:
        for cat in ['test']:
            datum = ds[cat].__getitem__(idx[cat])
            idx[cat] += 1
            K = datum['K']
            
            
            cam_scale = datum['cam_scale']
            rgb = datum['rgb'].transpose(1, 2, 0)[...,::-1].copy()# [...,::-1].copy()
            for i in range(22):
                pcld = datum['cld_rgb_nrm'][:3, :].transpose(1, 0).copy()
                p2ds = bs_utils.project_p3d(pcld, cam_scale, K)
                # rgb = bs_utils.draw_p2ds(rgb, p2ds)
                kp3d = datum['kp_3ds'][i]
                if kp3d.sum() < 1e-6:
                    break
                kp_2ds = bs_utils.project_p3d(kp3d, cam_scale, K)
                rgb = bs_utils.draw_p2ds(
                    rgb, kp_2ds, 3, bs_utils.get_label_color(datum['cls_ids'][i][0], mode=1)
                )
                ctr3d = datum['ctr_3ds'][i]
                ctr_2ds = bs_utils.project_p3d(ctr3d[None, :], cam_scale, K)
                rgb = bs_utils.draw_p2ds(
                    rgb, ctr_2ds, 4, (0, 0, 255)
                )
                imshow('{}_rgb'.format(cat), rgb)
                cmd = waitKey(0)
                if cmd == ord('q'):
                    exit()
                else:
                    continue


if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
