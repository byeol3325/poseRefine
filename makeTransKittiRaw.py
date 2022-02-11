kitti_path = '/data0/data/KITTI/monodepth/kitti_raw/2011_09_26/2011_09_26_drive_0001_sync/image_02/'
save_path = '/data0/moon/dsacstar/datasets/kitti_raw/train/'

imgs_path = kitti_path + 'data/'
K_path = kitti_path + 'cam.txt'
poses_path = kitti_path + 'poses.txt'

poses = open(poses_path)
poses = poses.readlines()

img_list = os.listdir(imgs_path)

theta = 0
phi = 0
gamma = 1

for i in range(len(img_list)):
    rot_img = warpImg(imgs_path + img_list[i], theta=theta, phi=phi, gamma=gamma)
    img = img_list[i]
    h, w, _ = rot_img.shape
    crop_image = rot_img[int(h/2-122):int(h/2+122), int(w/2-402):int(w/2+402), :]
    # save image
    cv2.imwrite(save_path + 'image/' + img[:-4] + "_" + str(theta) + "_" + str(phi) + "_" + str(gamma) + ".png", crop_image)

    # save poses
    poseL = poses[i].split()
    #print(poseL)
    poseL_elements = np.array([ [float(poseL[0]), float(poseL[1]), float(poseL[2]), float(poseL[3])],
                    [float(poseL[4]), float(poseL[5]), float(poseL[6]), float(poseL[7])],
                    [float(poseL[8]), float(poseL[9]), float(poseL[10]), float(poseL[11])] ])        
    
    rot3D = rotation3D(theta, phi, gamma)

    new_calib = np.matmul(rot3D[:3,:3], poseL_elements[:,:3])
    T = np.array([[poseL_elements[0,3]],[poseL_elements[1,3]],[poseL_elements[2,3]]])
    new_pose = np.concatenate((new_calib, T), axis=1)
    
    n_pose = poseL
    for i in range(len(n_pose)):
        n_pose[i] = str(new_pose[int(i/4)][i%4])
        
    n_pose = " ".join(n_pose)
    with open(save_path + "poses.txt", 'a') as file:
        file.writelines(n_pose + '\n')
