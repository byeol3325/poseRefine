import shutil
import os
import torch
import numpy as np
import cv2
import math

from torch.utils import data

import random

def get_rad(theta, phi, gamma):
    return (deg_to_rad(theta),
            deg_to_rad(phi),
            deg_to_rad(gamma))

def get_deg(rtheta, rphi, rgamma):
    return (rad_to_deg(rtheta),
            rad_to_deg(rphi),
            rad_to_deg(rgamma))

def deg_to_rad(deg):
    return deg * math.pi / 180.0

def rad_to_deg(rad):
    return rad * 180.0 / math.pi

def rotation3D(theta, phi, gamma):
    rtheta, rphi, rgamma = get_rad(theta, phi, gamma)

    RX = np.array([ [1, 0, 0, 0],
                        [0, np.cos(rtheta), -np.sin(rtheta), 0],
                        [0, -np.sin(rtheta), np.cos(rtheta), 0],
                        [0, 0, 0, 1]])
        
    RY = np.array([ [np.cos(rphi), 0, -np.sin(rphi), 0],
                        [0, 1, 0, 0],
                        [np.sin(rphi), 0, np.cos(rphi), 0],
                        [0, 0, 0, 1]])
    RZ = np.array([ [np.cos(rgamma), -np.sin(rgamma), 0, 0],
                        [np.sin(rgamma), np.cos(rgamma), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    R = np.dot(np.dot(RX, RY), RZ)

    return R

def genImage(ori_path, target_path, save_path, theta=0, phi=0, gamma=0):
    target_path_ = target_path + 'image_2/'
    save_path_rgb = save_path + 'rgb/'
    print('start')
    print('target_path_ path :', target_path_)
    print('save_path_rgb img path :', save_path_rgb)

    img_list = os.listdir(target_path_)
    for img in img_list:
        path_img = target_path_ + img
        target_img = save_path_rgb + img[:-4] + str(theta) + "_" + str(phi) + "_" + str(gamma) + ".png"
        shutil.copyfile(path_img, target_img)

    print('complete image')
    return None

def genCalib(ori_path, target_path, save_path, theta=0, phi=0, gamma=0):
    ori_path_calib = target_path + 'calib/'
    save_path_calib = save_path + 'calibration/'

    print('start')
    print('ori calib path :', ori_path_calib)
    print('save_path_calib path :', save_path_calib)

    calib_list = os.listdir(ori_path_calib)

    for calib in calib_list:
        calib_one = open(ori_path_calib + calib)
        lines = calib_one.readlines()

        P2 = lines[2].split()
        P2_elements = np.array ([ [float(P2[1]), float(P2[2]), float(P2[3]), float(P2[4])],
                        [float(P2[5]), float(P2[6]), float(P2[7]), float(P2[8])],
                        [float(P2[9]), float(P2[10]), float(P2[11]), float(P2[12])] ])
        
        rot3D = rotation3D(theta, phi, gamma)

        new_calib = np.matmul(rot3D[:3,:3], P2_elements)

        line_1 = str(new_calib[0][0])

        with open(save_path_calib + calib[:-4] + str(theta) + "_" + str(phi) + "_" + str(gamma) + ".txt", 'w') as file:
        #print(save_calib_dir + one)
            file.writelines(line_1)
    
    print('comple calib fin')
    return None 

def genPose(ori_path, target_path, save_path, theta=0, phi=0, gamma=0):
    ori_path_img = target_path + 'image_2/'
    ori_path_calib = target_path + 'calib/'
    target_path_img = save_path + 'rgb/'
    target_path_pose = save_path + 'poses/'

    print('start')
    print('ori calib path :', ori_path_calib)
    print('target pose path :', target_path_pose)

    img_list = os.listdir(ori_path_img)

    for img_file in img_list:
        ori_file_name = img_file[:-4]
        target_file_name = img_file[:-4] + str(theta) + "_" + str(phi) + "_" + str(gamma)

        ori_img = cv2.imread(ori_path_img + ori_file_name + ".png")
        target_img = cv2.imread(target_path_img + target_file_name + ".png")

        ori_calib = open(ori_path_calib + ori_file_name + ".txt", 'r')
        lines = ori_calib.readlines()
        P2 = lines[2].split()
        P2_elements = np.array ([ [float(P2[1]), float(P2[2]), float(P2[3]), float(P2[4])],
                        [float(P2[5]), float(P2[6]), float(P2[7]), float(P2[8])],
                        [float(P2[9]), float(P2[10]), float(P2[11]), float(P2[12])] ])

        ori_shape = ori_img.shape
        target_shape = target_img.shape

        c_h = ori_shape[0] - target_shape[0]
        c_w = ori_shape[1] - target_shape[1]
        P2_elements[0,2] = P2_elements[0,2] - c_w/2
        P2_elements[1,2] = P2_elements[1,2] - (c_h/2)

        focal_length = float(P2[1])

        cam_mat = torch.eye(3)
        cam_mat[0, 0] = focal_length
        cam_mat[1, 1] = focal_length
        cam_mat[0, 2] = ori_shape[1]/2
        cam_mat[1, 2] = ori_shape[0]/2
        cam_mat = cam_mat.numpy()
        R_T = np.matmul(np.linalg.inv(cam_mat), P2_elements)

        rot3D = rotation3D(theta, phi, gamma)

        R_T = np.matmul(rot3D[:3,:3], R_T)
    
        n = 4
        m = 4
        
        RT_all = [' '.join(str(x) for x in R_T[i,:]) for i in range(3)]
        RT_all.append('0 0 0 1')
        
        with open(target_path_pose + target_file_name + ".txt", 'w') as file:
            for i in range(4):
                file.writelines(RT_all[i])
                file.writelines('\n')
        
    print('comple pose fin')
    return None


########################################################
##                     example                        ##
########################################################
# original gen
"""
ori_path = '~/datasets/original/training/'
target_path = '~/datasets/original_gen/training/'
save_path = '~datasets/kitti_pose/train/'

genImage(ori_path, target_path, save_path, 0, 0, 0)
genCalib(ori_path, target_path, save_path, 0, 0, 0)
genPose(ori_path, target_path, save_path, 0, 0, 0)

"""

# rotation z
"""
deg_z = 0.1
ori_path = '~/datasets/original/training/'
target_path = '~/datasets/rotz_calib/rotz_calib/rotation' + str(deg_z) + '/fixed' + str(deg_z) + '/training/'
save_path = '~datasets/kitti_pose/train/'

genImage(ori_path, target_path, save_path, 0, 0, deg_z)
genCalib(ori_path, target_path, save_path, 0, 0, deg_z)
genPose(ori_path, target_path, save_path, 0, 0, deg_z)

"""


# rotation x (tilt) ***** NOT YET
"""
deg_x = 1
ori_path = '~/datasets/original/training/'
target_path = '~/datasets/rotz_calib/rotz_calib/rotation' + str(deg_z) + '/fixed' + str(deg_z) + '/training/'
save_path = '~datasets/kitti_pose/train/'

rotImage(ori_path, save_path, deg_x, 0, 0)
genCalib(ori_path, save_path, deg_x, 0, 0)
genPose(ori_path, save_path, deg_x, 0, 0)
"""

