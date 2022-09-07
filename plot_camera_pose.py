#import utils
#from checkpoints import sorted_ckpts
#from models.cam_params import CamParams
#from models.frameworks import create_model
#from dataio.kitti360 import KITTI360Dataset

import os
import sys
import torch
import numpy as np
from load_kitti import load_kitti

"""
modified from https://github.com/opencv/opencv/blob/master/samples/python/camera_calibration_show_extrinsics.py
"""

def inverse_homogeneoux_matrix(M):
    R = M[0:3, 0:3]
    T = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -(R.T).dot(T)

    return M_inv


def transform_to_matplotlib_frame(cMo, X, inverse=False):
    M = np.identity(4)
    M[1, 1] = 0
    M[1, 2] = 1
    M[2, 1] = -1
    M[2, 2] = 0

    if inverse:
        return M.dot(inverse_homogeneoux_matrix(cMo).dot(X))
    else:
        return M.dot(cMo.dot(X))


def create_camera_model(camera_matrix, width, height, scale_focal, draw_frame_axis=False):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    focal = 2 / (fx + fy)
    f_scale = scale_focal * focal

    # draw image plane
    X_img_plane = np.ones((4, 5))
    X_img_plane[0:3, 0] = [-width, height, f_scale]
    X_img_plane[0:3, 1] = [width, height, f_scale]
    X_img_plane[0:3, 2] = [width, -height, f_scale]
    X_img_plane[0:3, 3] = [-width, -height, f_scale]
    X_img_plane[0:3, 4] = [-width, height, f_scale]

    # draw triangle above the image plane
    X_triangle = np.ones((4, 3))
    X_triangle[0:3, 0] = [-width, -height, f_scale]
    X_triangle[0:3, 1] = [0, -2*height, f_scale]
    X_triangle[0:3, 2] = [width, -height, f_scale]

    # draw camera
    X_center1 = np.ones((4, 2))
    X_center1[0:3, 0] = [0, 0, 0]
    X_center1[0:3, 1] = [-width, height, f_scale]

    X_center2 = np.ones((4, 2))
    X_center2[0:3, 0] = [0, 0, 0]
    X_center2[0:3, 1] = [width, height, f_scale]

    X_center3 = np.ones((4, 2))
    X_center3[0:3, 0] = [0, 0, 0]
    X_center3[0:3, 1] = [width, -height, f_scale]

    X_center4 = np.ones((4, 2))
    X_center4[0:3, 0] = [0, 0, 0]
    X_center4[0:3, 1] = [-width, -height, f_scale]

    # draw camera frame axis
    X_frame1 = np.ones((4, 2))
    X_frame1[0:3, 0] = [0, 0, 0]
    X_frame1[0:3, 1] = [f_scale/2, 0, 0]

    X_frame2 = np.ones((4, 2))
    X_frame2[0:3, 0] = [0, 0, 0]
    X_frame2[0:3, 1] = [0, f_scale/2, 0]

    X_frame3 = np.ones((4, 2))
    X_frame3[0:3, 0] = [0, 0, 0]
    X_frame3[0:3, 1] = [0, 0, f_scale/2]

    if draw_frame_axis:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4, X_frame1, X_frame2, X_frame3]
    else:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]


def draw_camera(ax, camera_matrix, cam_width, cam_height, scale_focal, extrinsics, annotation=True):
    from matplotlib import cm

    min_values = np.zeros((3, 1))
    min_values = np.inf
    max_values = np.zeros((3, 1))
    max_values = -np.inf

    X_moving = create_camera_model(
        camera_matrix, cam_width, cam_height, scale_focal, False)

    cm_subsection = np.linspace(0.0, 1.0, extrinsics.shape[0])
    colors = [cm.jet(x) for x in cm_subsection]

    for idx in range(extrinsics.shape[0]):
        # R, _ = cv.Rodrigues(extrinsics[idx,0:3])
        # cMo = np.eye(4,4)
        # cMo[0:3,0:3] = R
        # cMo[0:3,3] = extrinsics[idx,3:6]
        cMo = extrinsics[idx]
        for i in range(len(X_moving)):
            X = np.zeros(X_moving[i].shape)
            for j in range(X_moving[i].shape[1]):
                X[0:4, j] = transform_to_matplotlib_frame(cMo, X_moving[i][0:4, j], True)
            ax.plot3D(X[0, :], X[1, :], X[2, :], color=colors[idx])
            min_values = np.minimum(min_values, X[0:3, :].min(1))
            max_values = np.maximum(max_values, X[0:3, :].max(1))
        # modified: add an annotation of number
        if annotation:
            X = transform_to_matplotlib_frame(cMo, X_moving[0][0:4, 0], True)
            ax.text(X[0], X[1], X[2], "{}".format(idx), color=colors[idx])

    return min_values, max_values

def get_test_block_final(args,dropout,block):
    assert dropout == 0.8 or dropout == 4.0
    train_frames = []
    test_frames = []
    split_file_train = os.path.join(args.data.data_root,'split_%.1f'%dropout,'train_%02d.txt'%block)
    with open(split_file_train) as f:
        lines_train = f.readlines()
    split_file_test = os.path.join(args.data.data_root,'split_%.1f'%dropout,'test_%02d.txt'%block)
    with open(split_file_test) as f:
        lines_test = f.readlines()

    for train_file in lines_train:
        train_file = train_file.split(' ')[0]
        curr_drive_seq_train = int(train_file.split('/')[1].split('_')[4])
        curr_frame_train = int(train_file.split('/')[-1][:-4])
        train_frames.append(curr_frame_train)

    for test_file in lines_test:
        test_file = test_file.split(' ')[0]
        curr_drive_seq_train = int(test_file.split('/')[1].split('_')[4])
        curr_frame_test = int(test_file.split('/')[-1][:-4])
        test_frames.append(curr_frame_test)

    block = [curr_drive_seq_train,train_frames,test_frames]
    return block

def main_function():#args,curr_block
    #--------------
    # parameters
    #--------------  
    cam_width = 0.0064 * 5 / 2 /2
    cam_height = 0.0048 * 5 / 2 /2
    scale_focal = 5.

    #--------------
    # Load model
    #--------------
    #device_ids = args.device_ids
    #device = "cuda:{}".format(device_ids[0])
    
    #drive_seq = curr_block[0]
    #num_imgs = [curr_block[1][0],block[1][-1]]
    #print('num_imgs ',num_imgs)
    #[self.ego_img_nums[ self.frame_range[0] ], self.ego_img_nums[ self.frame_range[1] ]]
    #ego_pose_file = os.path.join(args.data.data_root,'data_poses','2013_05_28_drive_' + str(drive_seq).rjust(4,'0') + '_sync', 'cam0_to_world.txt')
    #cam_pose_file = os.path.join(args.data.data_root,'calibration', 'calib_cam_to_pose.txt')
    #intrinsic_file = os.path.join(args.data.data_root,'calibration', 'perspective.txt')

    _, poses, _, test_poses, render_poses, random_poses, hwf, near, far, _ = load_kitti(
                                                        data_root = '/mnt/qb/work/geiger/abhattacharyya22/data/KITTI-360', 
                                                        mode = 'test', 
                                                        dropout= 0.8,
                                                        block = 2, 
                                                        coord_scale_factor=50.0,
                                                        so3_representation = 'quaternion', 
                                                        intrinsics_representation = 'square',
                                                        initial_fov = 53.13,
                                                        reg_pose_list = None)


    #--------------
    # Load camera parameters
    #--------------
    #cam_params = cam_params_left#CamParams.from_state_dict(state_dict['cam_param'])
    H = hwf[0]
    W = hwf[1]
    #c2ws_left = cam_params_left.get_camera2worlds().data.cpu().numpy()
    #c2ws_right = cam_params_right.get_camera2worlds().data.cpu().numpy()
    ## hack
    '''print('----hack for image2')
    t220 = np.load('t220.npy')
    print(t220)
    t220[:3, 3] /= 50.
    c2ws2 = c2ws @ t220[None]'''
    c2ws = poses#np.concatenate((c2ws_left, c2ws_right), 0)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-variable

    fig, axs = plt.subplots(3)

    top_tracks_l = poses[:poses.shape[0]//2,:3,3][:,[0,1]]
    top_tracks_r = poses[poses.shape[0]//2:,:3,3][:,[0,1]]
    axs[0].scatter(top_tracks_l[:,0],top_tracks_l[:,1],c='r')
    axs[0].scatter(top_tracks_r[:,0],top_tracks_r[:,1],c='g')

    top_tracks_l = poses[:poses.shape[0]//2,:3,3][:,[0,2]]
    top_tracks_r = poses[poses.shape[0]//2:,:3,3][:,[0,2]]
    axs[1].scatter(top_tracks_l[:,0],top_tracks_l[:,1],c='r')
    axs[1].scatter(top_tracks_r[:,0],top_tracks_r[:,1],c='g')

    top_tracks_l = poses[:poses.shape[0]//2,:3,3][:,[1,2]]
    top_tracks_r = poses[poses.shape[0]//2:,:3,3][:,[1,2]]
    axs[2].scatter(top_tracks_l[:,0],top_tracks_l[:,1],c='r')
    axs[2].scatter(top_tracks_r[:,0],top_tracks_r[:,1],c='g')
    plt.savefig('./cam.png')
    print('Done!')

    sys.exit(0)


    def get_intrinsic():
        fx, fy = hwf[-1], hwf[-1]
        intr = torch.eye(3)
        #cx = self.W0 / 2.
        #cy = self.H0 / 2.
        cx = 682.049453
        cy = 238.769549
        # OK with grad: with produce grad_fn=<CopySlices>
        intr[0, 0] = fx
        intr[1, 1] = fy
        intr[0, 2] = cx
        intr[1, 2] = cy
        return intr

    intr = get_intrinsic().data.cpu().numpy()
    extrinsics = np.linalg.inv(c2ws)

    #import pdb#; pdb.set_trace()
    '''T = extrinsics
    print(T[7])
    print(T[8])
    print(T[7+16])
    print(T[8+16])
    T = extrinsics[:, :3, 3]
    print(T[8])
    print(T[16+8])
    diffnorm = np.linalg.norm(T[:-1, :] - T[1:, :], axis=1)
    print(diffnorm)
    print(intr)'''
    #--------------
    # Draw cameras
    #--------------

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.set_aspect("equal")
    # ax.set_aspect("auto")

    min_values, max_values = draw_camera(ax, intr, cam_width, cam_height, scale_focal, extrinsics[:], False)

    '''z = np.arange(-0.9, 0.91, 0.01)
    x = np.arange(-0.04,0.045,0.005)
    i,j = np.meshgrid(x,z)
    i = i.reshape(-1)[:,None]
    j = j.reshape(-1)[:,None]
    k = np.ones((len(i),1))
    feat_pos = np.concatenate([i,j,k],axis=-1)

    ax.scatter(feat_pos[:,0], feat_pos[:,1], feat_pos[:,2], marker='o',c='r')'''

    X_min = min_values[0]
    X_max = max_values[0]
    Y_min = min_values[1]
    Y_max = max_values[1]
    Z_min = min_values[2]
    Z_max = max_values[2]
    max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0
    print('max_range ',max_range)

    mid_x = (X_max+X_min) * 0.5
    mid_y = (Y_max+Y_min) * 0.5
    mid_z = (Z_max+Z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range )
    ax.view_init(azim=180)


    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')
    ax.set_title('Camera Trajectory')

    plt.savefig('./cam.png')
    print('Done!')


if __name__ == "__main__":
    # Arguments
    #parser = utils.create_args_parser()
    #args, unknown = parser.parse_known_args()
    #config = utils.load_config(args, unknown)
    #block = get_test_block_final(config,args.dropout,args.test_block)
    main_function()
