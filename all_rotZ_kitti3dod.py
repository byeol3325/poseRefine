### only change gamma(Z)
### i will update theta(X)


import os
import math
import cv2
import numpy as np
import imutils


def cropRotImage(ori_path, new_path, train=True, theta=0, phi=0, gamma=0):
    if (train == False):
        ori_path += 'testing/'
        new_path += 'testing/'
    else:
        ori_path += 'training/'
        new_path += 'training/'
    
    #ori_image_pathR = ori_path + 'image_3/'
    #new_image_pathR = new_path + 'image_3/'

    ori_image_pathL = ori_path + 'image_2/'
    #new_image_pathL = new_path + 'image_2/'
    if os.path.isdir(new_path) == False:
        os.mkdir(new_path)

    print(new_path, ' start.')

    original_calib = ori_path + 'calib/'
    original_label = ori_path + 'label_2/'
    
    file_list = os.listdir(ori_image_pathL)
    file_list_py = [file for file in file_list if file.endswith('.png')]
    
    new_path_img = new_path + 'rgb/'
    new_path_poses = new_path + 'poses/'
    new_path_calibration = new_path + 'calibration/'
    new_path_label = new_path + 'label_2/'
    new_path_calib = new_path + 'calib/'
    new_path_cam = new_path + 'cam/'

    for file_name in file_list_py:
        #new_path_one = new_path + file_name[:-4] + '/'
        #print("CHECK PATH :", new_path_one)
        
        #if os.path.isdir(new_path_one) == False:
        #    os.mkdir(new_path_one)

        imgL = cv2.imread(ori_image_pathL + file_name)
        #imgR = cv2.imread(ori_image_pathR + file_name)

        rotate_bound = imutils.rotate_bound(imgL, gamma)
        r_h, r_w, _ = rotate_bound.shape
        h, w, _ = imgL.shape
        pi_angle = math.pi/180*gamma
        ratio = h * math.sqrt(1 + math.tan(pi_angle)**2) / 2 / (h+w*math.tan(pi_angle))
        new_h = h*ratio
        new_w = w*ratio
        cropped_imgL = rotate_bound[int(-new_h + r_h/2):int(new_h + r_h/2), int(-new_w + r_w/2):int(new_w + r_w/2), :]
        #cv2.imwrite(new_image_path + file_name, cropped_img)
        c_h, c_w, _ = cropped_imgL.shape
        fixed_imgL = cropped_imgL[int(c_h/2-122):int(c_h/2+122),int(c_w/2-402):int(c_w/2+402),:]
        cv2.imwrite(new_path_img + file_name[:-4] + "_" + str(theta) + "_" + str(phi) + "_" + str(gamma) + ".png", fixed_imgL)

        #print(size_image_path + file_name)
        #cv2.imwrite(new_path_one + '0000.png', fixed_imgL)
        #print(new_image_path + file_name)
        #print(size_image_path + file_name)
        """
        rotate_bound = imutils.rotate_bound(imgR, angle)
        r_h, r_w, _ = rotate_bound.shape
        h, w, _ = imgR.shape
        pi_angle = math.pi/180*angle
        ratio = h * math.sqrt(1 + math.tan(pi_angle)**2) / 2 / (h+w*math.tan(pi_angle))
        new_h = h*ratio
        new_w = w*ratio
        cropped_imgR = rotate_bound[int(-new_h + r_h/2):int(new_h + r_h/2), int(-new_w + r_w/2):int(new_w + r_w/2), :]
        #cv2.imwrite(new_image_path + file_name, cropped_img)
        c_h, c_w, _ = cropped_imgR.shape
        fixed_imgR = cropped_imgR[int(c_h/2-122):int(c_h/2+122),int(c_w/2-402):int(c_w/2+402),:]
        #print(size_image_path + file_name)
        cv2.imwrite(new_path_one + '0001.png', fixed_imgR)
        """
        calib_one = open(original_calib + file_name[:-4] + '.txt')
        lines = calib_one.readlines()

        pi_angle = math.pi/180*gamma
        s = math.sin(pi_angle)
        c = math.cos(pi_angle)
        rotz = np.array([ [c, -s, 0], [s, c, 0], [0, 0, 1] ])

        P2 = lines[2].split()
        new_P2 = P2
        P2_elements = np.array ([ [float(P2[1]), float(P2[2]), float(P2[3]), float(P2[4])],
                        [float(P2[5]), float(P2[6]), float(P2[7]), float(P2[8])],
                        [float(P2[9]), float(P2[10]), float(P2[11]), float(P2[12])] ])
        
        c_u = P2_elements[0,2]
        c_v = P2_elements[1,2]
        f_u = P2_elements[0,0]
        f_v = P2_elements[1,1]
        b_x = P2_elements[0,3]/f_u
        b_y = P2_elements[1,3]/f_v
        b_z = P2_elements[2,3]

        K = np.array([ [f_u, 0, c_u], 
              [0, f_v, c_v], 
              [0, 0, 1]
             ])

        ori_shapeL = imgL.shape
        cropped_shapeL = cropped_imgL.shape
        new_shapeL = fixed_imgL.shape

        u = ori_shapeL[1]/2 - c_u
        v = ori_shapeL[0]/2 - c_v
        new_c_u = cropped_shapeL[1]/2 - u
        new_c_v = cropped_shapeL[0]/2 - v

        print(K)
        K = np.array([ [f_u, 0, new_c_u], 
                     [0, f_v, new_c_v], 
                     [0, 0, 1]
                    ])
        print(K)
        R = np.eye(3)
        T = np.array([[b_x], [b_y], [b_z]])
        rotationR = np.matmul(rotz, R)
        newRTL = np.concatenate((rotationR, T), axis=1)
        # P2_elements = np.matmul(K, newRT)

        c_h = cropped_shapeL[0] - new_shapeL[0]
        c_w = cropped_shapeL[1] - new_shapeL[1]
        K[0,2] = K[0,2] - c_w/2
        K[1,2] = K[1,2] - c_h/2
        print(K)
        focal = P2[1]
        
        with open(new_path_calibration + file_name[:-4] + "_" + str(theta) + "_" + str(phi) + "_" + str(gamma) + ".txt", 'w') as file:
            file.writelines(focal)
        
        P2_elements = np.matmul(K, newRTL)
        
        cam = [str(K[0,0]) + " " + str(K[0,1]) + " " + str(K[0,2]) + '\n',
               str(K[1,0]) + " " + str(K[1,1]) + " " + str(K[1,2]) + '\n',
               str(K[2,0]) + " " + str(K[2,1]) + " " + str(K[2,2])]
        
        with open(new_path_cam+ file_name[:-4] + "_" + str(theta) + "_" + str(phi) + "_" + str(gamma) + ".txt", 'w') as file:
            file.writelines(cam)
        
        for i in range(len(new_P2)-1):
            new_P2[i+1] = str(P2_elements[int(i/4),i%4])
        new_P2 = " ".join(new_P2)
        lines[2] = new_P2
        with open(new_path_calib + file_name[:-4] + "_" + str(theta) + "_" + str(phi) + "_" + str(gamma) + ".txt", 'w') as file:
            file.writelines(lines[:3])
            file.writelines('\n')
            file.writelines(lines[3:])
        

        """
        P3 = lines[3].split()
        new_P3 = P3
        P3_elements = np.array ([ [float(P3[1]), float(P3[2]), float(P3[3]), float(P3[4])],
                        [float(P3[5]), float(P3[6]), float(P3[7]), float(P3[8])],
                        [float(P3[9]), float(P3[10]), float(P3[11]), float(P3[12])] ])
        c_u = P3_elements[0,2]
        c_v = P3_elements[1,2]
        f_u = P3_elements[0,0]
        f_v = P3_elements[1,1]
        b_x = P3_elements[0,3]/f_u
        b_y = P3_elements[1,3]/f_v
        b_z = P3_elements[2,3]
        R = np.eye(3)
        T = np.array([[b_x], [b_y], [b_z]])
        rotationR = np.matmul(rotz, R)
        newRTR = np.concatenate((rotationR, T), axis=1)
        poses = [str(newRTL[0,0]) + " " + str(newRTL[0,1]) + " " + str(newRTL[0,2]) + " " + str(newRTL[0,3]) + " " +
               str(newRTL[1,0]) + " " + str(newRTL[1,1]) + " " + str(newRTL[1,2]) + " " + str(newRTL[1,3]) + " " +
               str(newRTL[2,0]) + " " + str(newRTL[2,1]) + " " + str(newRTL[2,2]) + " " + str(newRTL[2,3]) + '\n',
               str(newRTR[0,0]) + " " + str(newRTR[0,1]) + " " + str(newRTR[0,2]) + " " + str(newRTR[0,3]) + " " +
               str(newRTR[1,0]) + " " + str(newRTR[1,1]) + " " + str(newRTR[1,2]) + " " + str(newRTR[1,3]) + " " +
               str(newRTR[2,0]) + " " + str(newRTR[2,1]) + " " + str(newRTR[2,2]) + " " + str(newRTR[2,3])]
        
        with open(new_path_one + 'poses.txt', 'w') as file:
            file.writelines(poses)
        """
        
        poses = [' '.join(str(x) for x in newRTL[i,:]) for i in range(3)]
        poses.append('0 0 0 1')
        with open(new_path_poses + file_name[:-4] + "_" + str(theta) + "_" + str(phi) + "_" + str(gamma) + ".txt", 'w') as file:
            for i in range(4):
                file.writelines(poses[i])
                file.writelines('\n')

        label_one = open(original_label + file_name[:-4] + '.txt')
        label_lines = label_one.readlines()
        size_lines = label_lines

        pi_angle = -math.pi/180*gamma
        rotMatrix = np.array([[np.cos(pi_angle), -np.sin(pi_angle)],
                     [np.sin(pi_angle), np.cos(pi_angle)]])

        c_h = ori_shapeL[0] - new_shapeL[0]
        c_w = ori_shapeL[1] - new_shapeL[1]

        for i_th in range(len(label_lines)):
            elements = label_lines[i_th].split()
            
            a1 = np.array([ -ori_shapeL[1]/2+float(elements[4]), ori_shapeL[0]/2-float(elements[5]) ]) # 712, 143 => 546 112
            a2 = np.array([ -ori_shapeL[1]/2+float(elements[4]), ori_shapeL[0]/2-float(elements[7]) ])
            a3 = np.array([ -ori_shapeL[1]/2+float(elements[6]), ori_shapeL[0]/2-float(elements[5]) ])
            a4 = np.array([ -ori_shapeL[1]/2+float(elements[6]), ori_shapeL[0]/2-float(elements[7]) ])

            p1 = np.matmul(rotMatrix, a1)
            p2 = np.matmul(rotMatrix, a2)
            p3 = np.matmul(rotMatrix, a3)
            p4 = np.matmul(rotMatrix, a4)
    
            new_p1 = np.array([ p1[0]+ori_shapeL[1]/2, -p1[1]+ori_shapeL[0]/2])
            new_p2 = np.array([ p2[0]+ori_shapeL[1]/2, -p2[1]+ori_shapeL[0]/2])
            new_p3 = np.array([ p3[0]+ori_shapeL[1]/2, -p3[1]+ori_shapeL[0]/2])
            new_p4 = np.array([ p4[0]+ori_shapeL[1]/2, -p4[1]+ori_shapeL[0]/2])
            min_w = min(new_p1[0], new_p2[0], new_p3[0], new_p4[0])
            max_w = max(new_p1[0], new_p2[0], new_p3[0], new_p4[0])
            min_h = min(new_p1[1], new_p2[1], new_p3[1], new_p4[1])
            max_h = max(new_p1[1], new_p2[1], new_p3[1], new_p4[1])
    
            elements[4] = str(min_w)
            elements[6] = str(max_w)
            elements[5] = str(min_h)
            elements[7] = str(max_h)
    
            new_line = " ".join(elements)
            label_lines[i_th] = new_line
            #size_lines = lines
            size_elements = label_lines[i_th].split()
            size_elements[4] = str(float(size_elements[4]) - c_w/2)
            size_elements[5] = str(float(size_elements[5]) - c_h/2)
            size_elements[6] = str(float(size_elements[6]) - c_w/2)
            size_elements[7] = str(float(size_elements[7]) - c_h/2)

            xmin = float(size_elements[4])
            ymin = float(size_elements[5])
            xmax = float(size_elements[6])
            ymax = float(size_elements[7])
            if xmin < 0 or ymin < 0 or xmax > new_shapeL[1] or ymax > new_shapeL[0]:
                area = (xmax - xmin) * (ymax - ymin)
                new_xmin = xmin
                new_ymin = ymin
                new_xmax = xmax
                new_ymax = ymax
                if xmin < 0:
                    new_xmin = 0
                if ymin < 0:
                    new_ymin = 0
                if xmax > new_shapeL[1]:
                    new_xmax = new_shapeL[1]
                if ymax > new_shapeL[0]:
                    new_ymax = new_shapeL[0]
                new_area = (new_xmax - new_xmin) * (new_ymax - new_ymin)
                if new_area <= 0.3 * area:
                    size_elements[0] = 'DontCare'

            size_line = " ".join(size_elements)
            size_lines[i_th] = size_line
        with open(new_path_label + file_name[:-4] + "_" + str(theta) + "_" + str(phi) + "_" + str(gamma) + ".txt", 'w') as file:
            for i in range(len(size_lines)):
                file.writelines(size_lines[i])
                file.writelines('\n')
            
            
    print(gamma, ' angle image is fin')

    return None

ori_path = '/data0/data/KITTI/3Dobject_one/'
new_path = '/data0/data/KITTI/3Dobject_one/temp/'
cropRotImage(ori_path, new_path, train=True, theta=0, phi=0, gamma=0)
#cropRotImage(ori_path, new_path, train=True, theta=0, phi=0, gamma=1)
#cropRotImage(ori_path, new_path, train=True, theta=0, phi=0, gamma=2)
#cropRotImage(ori_path, new_path, train=True, theta=0, phi=0, gamma=3)
#cropRotImage(ori_path, new_path, train=True, theta=0, phi=0, gamma=-1)
#cropRotImage(ori_path, new_path, train=True, theta=0, phi=0, gamma=-2)
#cropRotImage(ori_path, new_path, train=True, theta=0, phi=0, gamma=-3)

#### change gamma
#### |new_path/
####   |image_2/                 # RGB Image
####   |label_2/                 # 2D, 3D labels for 3DOD
####   |calib/                   # K[R|T], projection matrix
####   |rgb/                     # RGB Image
####   |poses/                   # R|T, rotation & translation matrix
####   |calibration/             # focal length
####   |cam/                     # K, intrinsic matrix

