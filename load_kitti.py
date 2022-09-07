import os
import sys
import glob
import re
import random
import itertools
import pickle as pkl
import gc
import numpy as np
#import cv2
#from pyntcloud import PyntCloud
from PIL import Image, ImageDraw, ImageFont
import scipy.ndimage as snd
#import imageio
#import skimage
#from skimage.restoration import inpaint
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pytorch3d import transforms as tr3d
import torch.nn.functional as F

from cam_params import CamParams, interpolate_poses

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d, griddata
from scipy.spatial.transform import Rotation as R

from run_nerf_helpers import *

#from utils import get_bbox3d_for_kitti

'''def lr_consistency(disp_left,disp_right):
    #print('disp_left,disp_right ',disp_left.shape,disp_right.shape)
    for i in range(disp_left.shape[0]):
        for j in range(disp_left.shape[1]):
            j_dx = j - int(disp_left[i,j])
            j_dx = max(0,min(j_dx,disp_left.shape[1]-1))
            j_dx_right = disp_right[i,j_dx]
            final_j = j_dx + j_dx_right

            if abs(j - final_j) < 2:#disp_right[i,j_dx] != disp_left[i,j]:
                disp_left[i,j] = -1
                disp_right[i,j_dx] = -1
    return disp_left, disp_right'''

def lr_consistency(dleft,dright,left=True):
    H, W = dleft.shape
    out = np.zeros((H, W), dtype=np.uint8)#, len(thresholds)
    for i in range(H):
        for j in range(W):
            dleft_ij = dleft[i, j]
            posright_ij = [i, j - dleft_ij] if left else [i, j + dleft_ij]
            posright_ij[1] = max(0,min(posright_ij[1],W-1))
            
            dright_ij = dright[int(posright_ij[0]), int(posright_ij[1])]
            posleft_ij = [int(posright_ij[0]), int(posright_ij[1] + dright_ij)] \
                         if left else [int(posright_ij[0]), int(posright_ij[1] - dright_ij)]

            derror = abs(j - posleft_ij[1])
            #for thre_i, thre in enumerate(thresholds):
            out[i, j] = 1 if (derror <= 1) else 0
    return out


def lr_check_all(depths):
    num_pairs = len(depths) // 2
    left_mask, right_mask = [], []
    for i in tqdm(range(num_pairs)):
        disp_left = depths[i]
        disp_right = depths[i + (len(depths) // 2)]
        disp_left_mask = lr_consistency(disp_left.copy(),disp_right.copy())
        disp_right_mask = lr_consistency(disp_right.copy(),disp_left.copy(),left=False)

        '''fig, axs = plt.subplots(2,2)
        axs[0,0].imshow((0.60 * 552.554261 / (1 * disp_left)))
        axs[1,0].imshow((0.60 * 552.554261 / (1 * disp_right)))

        #_disp_left, _disp_right = lr_consistency(disp_left.copy(),disp_right.copy())
        
        #_disp_left[_disp_left < 0] = 1e-5
        #_disp_right[_disp_right < 0] = 1e-5
        #_disp_left = (0.60 * 552.554261 / (1 * (_disp_left + 1e-5)))
        #_disp_right = (0.60 * 552.554261 / (1 * (_disp_right + 1e-5)))
        
        
        axs[0,1].imshow(disp_left_mask)
        axs[1,1].imshow(disp_right_mask)

        plt.savefig('out.png')
        sys.exit(0)'''
        left_mask.append(disp_left_mask.copy())
        right_mask.append(disp_right_mask.copy())
    return left_mask + right_mask

 
def get_test_block_final(data_root,dropout,block):
    assert dropout == 0.8 or dropout == 4.0
    train_frames = []
    test_frames = []
    split_file_train = os.path.join(data_root,'split_%.1f'%dropout,'train_%02d.txt'%block)
    with open(split_file_train) as f:
        lines_train = f.readlines()
    split_file_test = os.path.join(data_root,'split_%.1f'%dropout,'test_%02d.txt'%block)
    with open(split_file_test) as f:
        lines_test = f.readlines()

    for train_file in lines_train:
        train_file = train_file.split(' ')[0]
        curr_drive_seq_train = int(train_file.split('/')[1].split('_')[4])
        curr_frame_train = int(train_file.split('/')[-1][:-4])
        train_frames.append(curr_frame_train)

    #print('train_frames ',train_frames)
    #sys.exit(0)

    for test_file in lines_test:
        test_file = test_file.split(' ')[0]
        curr_drive_seq_train = int(test_file.split('/')[1].split('_')[4])
        curr_frame_test = int(test_file.split('/')[-1][:-4])
        test_frames.append(curr_frame_test)

    block = [curr_drive_seq_train,train_frames,test_frames]
    return block

def load_kitti(data_root,mode,dropout,block,coord_scale_factor,so3_representation,
    intrinsics_representation,initial_fov,reg_pose_list,test_video=True,lr_check=False):

    #if mode == 'test':
    drive_seq,train_frames,eval_frames = get_test_block_final(data_root,dropout,block)
    print(drive_seq,train_frames,eval_frames)
    #else:
    #    drive_seq = 0
    #    train_frames = [6024 + 2*i for i in range(42)]
    #    eval_frames = [6024 + 42 + 2*i for i in range(8)]
    #sys.exit(0)
    #drive_seq = 8
    #train_frames = [845,847,849]
    #eval_frames = [846,848]


    frame_range = [train_frames[0],train_frames[-1]]
    print('frame_range ',frame_range)

    H = 376
    W = 1408
    near = 2.0
    far = 165.0
    scene_box = np.array([[ -34.3491, -8.15422, -40.5304 ],[ 38.5536, 30.0,
        111.638 ]])
    coord_scale_factor = (scene_box[1] - scene_box[0]) / 2.

    drive_seq = drive_seq
    frame_range = frame_range
    #PCDROOT = os.path.join(data_root,'data_3d_semantics','2013_05_28_drive_' + str(drive_seq).rjust(4,'0') + '_sync','static')

    #INSTROOT = os.path.join(data_root,'data_2d_semantics','train','2013_05_28_drive_' + str(drive_seq).rjust(4,'0') + '_sync','instance')
    im_data_loc = 'data_2d_raw' if mode == 'train' else 'data_2d_test'
    IMGROOTL = os.path.join(data_root,im_data_loc,'2013_05_28_drive_' + str(drive_seq).rjust(4,'0') + '_sync', 'image_00','data_rect')
    IMGROOTR = os.path.join(data_root,im_data_loc,'2013_05_28_drive_' + str(drive_seq).rjust(4,'0') + '_sync', 'image_01','data_rect')
    #IMGROOTFL = os.path.join(data_root,'data_2d_raw','2013_05_28_drive_' + str(drive_seq).rjust(4,'0') + '_sync', 'image_02','data_rgb')
    #IMGROOTFR = os.path.join(data_root,'data_2d_raw','2013_05_28_drive_' + str(drive_seq).rjust(4,'0') + '_sync', 'image_03','data_rgb')

    #ego_pose_file = os.path.join(data_root,'data_poses','2013_05_28_drive_' + str(drive_seq).rjust(4,'0') + '_sync', 'poses.txt')
    ego_pose_file = os.path.join(data_root,'data_poses','2013_05_28_drive_' + str(drive_seq).rjust(4,'0') + '_sync', 'cam0_to_world.txt')
    ego_poses = np.loadtxt(ego_pose_file)#delimiter=' '
    #print('ego_poses ',ego_poses.shape)
    #ego_traj = ego_poses[:,1:].reshape((-1,3,4))[::1,:,3][:,:2]
    ego_img_nums = ego_poses[:,0]
    ego_poses = ego_poses[:,1:].reshape((-1,4,4))

    STDROOTL = os.path.join(data_root,im_data_loc,'2013_05_28_drive_' + str(drive_seq).rjust(4,'0') + '_sync', 
                        'image_00','SMD-results')#-left-lrc
    STDROOTR = os.path.join(data_root,im_data_loc,'2013_05_28_drive_' + str(drive_seq).rjust(4,'0') + '_sync', 
                        'image_00','SMD-results-right')#-lrc

    STDROOTFULLL = os.path.join(data_root,'data_2d_raw','2013_05_28_drive_' + str(drive_seq).rjust(4,'0') + '_sync', 
                        'image_00','SMD-results-full-left')#-left-lrc
    STDROOTFULLR = os.path.join(data_root,'data_2d_raw','2013_05_28_drive_' + str(drive_seq).rjust(4,'0') + '_sync', 
                        'image_00','SMD-results-full-right')#-lrc

    num_imgs = [frame_range[0],frame_range[-1]]
    print('num_imgs ',num_imgs)
    #[ego_img_nums[ frame_range[0] ], ego_img_nums[ frame_range[1] ]]
    cam_pose_file = os.path.join(data_root,'calibration', 'calib_cam_to_pose.txt')
    intrinsic_file = os.path.join(data_root,'calibration', 'perspective.txt')
    cam_params_left, cam_params_right, pose_transform = CamParams.from_file(
        pose_file = ego_pose_file,
        cam_pose_file = cam_pose_file,
        intrinsic_file = intrinsic_file,
        num_imgs=num_imgs,  H0=H, W0=W,  
        so3_repr=so3_representation, intr_repr=intrinsics_representation, 
        coord_scale_factor=coord_scale_factor, initial_fov=initial_fov)
    cam_params_left = cam_params_left.cpu()
    cam_params_right = cam_params_right.cpu()

    pose_file = os.path.join(data_root,'video_poses','video_00.txt')
    cam_params_video, video_len = CamParams.transform_pose_list(pose_file,pose_transform,
            H0=H, W0=W,  
            so3_repr='quaternion', intr_repr='square', 
            coord_scale_factor=coord_scale_factor, initial_fov=53.13)


    images = []
    depths = []
    cams = []
    test_poses = []
    render_poses = []
    for index in tqdm(range(frame_range[1] - frame_range[0] + 1)):
        img_num_curr = frame_range[0] + index#ego_img_nums[ frame_range[0] + index ]
        #print('img_num_curr ',img_num_curr)
        if img_num_curr in train_frames:
            #if os.path.isfile(os.path.join(IMGROOTL,str(int(img_num_curr)).rjust(10,'0') + '.png')):
            base_im = Image.open(os.path.join(IMGROOTL,str(int(img_num_curr)).rjust(10,'0') + '.png'))
            base_im = np.array(base_im)/255.
            depth_im = np.load( os.path.join(STDROOTL, str(int(img_num_curr)).rjust(10,'0') + '_disp.npy') )
            #depth_im = (0.60 * 552.554261 / (1 * depth_im))
            #depth_im = depth_im/coord_scale_factor
            depths.append(depth_im)
        

            R, t, fx, fy = cam_params_left(index)
            tmat = np.eye(4)
            tmat[:3,:3] = R.detach().cpu().numpy()
            tmat[:3,3] = t.detach().cpu().numpy()
            cams.append(tmat.astype(np.float32))
            images.append(base_im)

    for index in tqdm(range(frame_range[1] - frame_range[0] + 1)):
        img_num_curr = frame_range[0] + index#ego_img_nums[ frame_range[0] + index ]
        #print('img_num_curr ',img_num_curr)
        if img_num_curr in train_frames:
            #if os.path.isfile(os.path.join(IMGROOTL,str(int(img_num_curr)).rjust(10,'0') + '.png')):
            base_im = Image.open(os.path.join(IMGROOTR,str(int(img_num_curr)).rjust(10,'0') + '.png'))
            base_im = np.array(base_im)/255.
            depth_im = np.load( os.path.join(STDROOTR, str(int(img_num_curr)).rjust(10,'0') + '_disp.npy') )
            #depth_im = (0.60 * 552.554261 / (1 * depth_im))
            #depth_im = depth_im/coord_scale_factor
            depths.append(depth_im)
        

            R, t, fx, fy = cam_params_right(index)
            tmat = np.eye(4)
            tmat[:3,:3] = R.detach().cpu().numpy()
            tmat[:3,3] = t.detach().cpu().numpy()
            cams.append(tmat.astype(np.float32))
            images.append(base_im)

    '''fig, axs = plt.subplots(2,2)
    axs[0,0].imshow(images[0])
    axs[1,0].imshow(images[1])

    #_disp_left, _disp_right = lr_consistency(disp_left.copy(),disp_right.copy())
    
    #_disp_left[_disp_left < 0] = 1e-5
    #_disp_right[_disp_right < 0] = 1e-5
    #_disp_left = (0.60 * 552.554261 / (1 * (_disp_left + 1e-5)))
    #_disp_right = (0.60 * 552.554261 / (1 * (_disp_right + 1e-5)))
    
    
    axs[0,1].imshow(images[2])
    axs[1,1].imshow(images[3])

    plt.savefig('out.png')
    sys.exit(0)'''

    for video_im_num in range(video_len):
        R, t, fx, fy = cam_params_video(video_im_num)
        tmat = np.eye(4)
        tmat[:3,:3] = R.detach().cpu().numpy()#rot_perturb[video_im_num] @ 
        tmat[:3,3] = t.detach().cpu().numpy()
        render_poses.append(tmat.astype(np.float32))
    
    for index in tqdm(range(frame_range[1] - frame_range[0])):
        img_num_curr = frame_range[0] + index#ego_img_nums[ frame_range[0] + index ]
        if img_num_curr in eval_frames:
            R, t, fx, fy = cam_params_left(index)
            tmat = np.eye(4)
            tmat[:3,:3] = R.detach().cpu().numpy()
            tmat[:3,3] = t.detach().cpu().numpy()
            test_poses.append(tmat.astype(np.float32))

    images = np.array(images)
    if lr_check:
        valid_masks = lr_check_all(depths)
        checked_depths = []
        for depth_im, mask in zip(depths,valid_masks):
            depth_im = (0.60 * 552.554261) / depth_im
            depth_im = depth_im/coord_scale_factor
            depth_im[mask == 0] = 0
            checked_depths.append(depth_im)
        depths = np.array(checked_depths)
    else:
        depths = np.array(depths)
        depths = (0.60 * 552.554261) / depths
        #depths = depths/coord_scale_factor
    
    poses = np.array(cams)
    test_poses = np.array(test_poses)
    render_poses = np.array(render_poses)
    hwf = [H,W,float(fx.detach().cpu().numpy())]

    #bboxs = get_bbox3d_for_kitti(poses, hwf)
    #print('images, poses ',images.shape,poses.shape,render_poses.shape)
    #sys.exit(0)
    if reg_pose_list is not None:
        reg_poses = sample_poses( reg_pose_list, pose_transform, hwf, coord_scale_factor )
    else:
        reg_poses = interpolate_poses( poses )

    return images, poses, depths, test_poses, render_poses, reg_poses, hwf, near, far, pose_transform


def sample_poses( pose_file, pose_transform, hwf, coord_scale_factor ):
    poses = np.loadtxt(pose_file, delimiter=' ')
    poses = poses.reshape(-1,4,4)
    poses = pose_transform[None] @ poses # render_poses
    poses[:, :3, 3] = poses[:, :3, 3] / coord_scale_factor
    return poses
    '''sample_poses = []
    for idx in range(20,100):
        cam_params_video, video_len = CamParams.transform_pose_list(pose_file,pose_transform,
                    center_idx = idx,
                    H0=hwf[0], W0=hwf[1],  
                    so3_repr='quaternion', intr_repr='square', 
                    coord_scale_factor=50., initial_fov=53.13)
        for video_im_num in range(video_len):
            R, t, fx, fy = cam_params_video(video_im_num)
            tmat = np.eye(4)
            tmat[:3,:3] = R.detach().cpu().numpy()#rot_perturb[video_im_num] @ 
            tmat[:3,3] = t.detach().cpu().numpy()
            sample_poses.append(tmat.astype(np.float32))'''
    '''sample_poses = []
    cam_params_video, video_len = CamParams.transform_pose_list(pose_file,pose_transform,
                    center_idx = -1,
                    H0=hwf[0], W0=hwf[1],  
                    so3_repr='quaternion', intr_repr='square', 
                    coord_scale_factor=50., initial_fov=53.13)
    for video_im_num in range(video_len):
        R, t, fx, fy = cam_params_video(video_im_num)
        tmat = np.eye(4)
        tmat[:3,:3] = R.detach().cpu().numpy()#rot_perturb[video_im_num] @ 
        tmat[:3,3] = t.detach().cpu().numpy()
        sample_poses.append(tmat.astype(np.float32))
    return np.array(sample_poses)'''

def project_points_to_image( projection_points, projection_depths, reference_im, target_depths ):
    projection_image = np.zeros((reference_im.shape[0],
        reference_im.shape[1],
        reference_im.shape[2]+1)
    )
    H, W = reference_im.shape[0], reference_im.shape[1]
    projection_image[:,:,3] = float('inf')
    projection_points = projection_points.astype(np.int32)
    valid_mask = np.zeros((H,W)).astype(np.bool)

    '''target_depths_i, target_depths_j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    target_depth_points = np.stack([target_depths_i, target_depths_j], -1)
    proj_depths = griddata(target_depth_points.reshape(-1,2), target_depths.reshape(-1), projection_points.reshape(-1,2))
    valid_mask = (proj_depths == projection_depths.reshape(-1)).reshape((H,W))'''

    for i in range(H):
        for j in range(W):
            proj_i = projection_points[i,j,0]
            proj_j = projection_points[i,j,1]

            if (0 <= proj_i < H) and (0 <= proj_j < W):
                proj_d = projection_depths[i,j]
                curr_d = projection_image[projection_points[i,j,0],projection_points[i,j,1],3]
                if curr_d > proj_d:
                    projection_image[proj_i,proj_j,:3] = reference_im[i,j,:]
                    projection_image[proj_i,proj_j,3] = curr_d

                true_depth = target_depths[proj_i,proj_j]
                #if not valid_mask[i,j]:
                if np.abs(true_depth - proj_d) < 0.025:
                    valid_mask[i,j] = True
    projection_image = None
    return projection_image, valid_mask

#images, rays, poses, depths, masks

def project_im(reference_rays, reference_depths, reference_im, target_pose, target_intrinsic, target_depths, H, W):
    reference_rays_o, reference_rays_d = reference_rays[0],  reference_rays[1]
    reference_rays_o = reference_rays_o.reshape(-1,3)
    reference_rays_d = reference_rays_d.reshape(-1,3)

    reference_depths = reference_depths.reshape(-1)
    reference_points = reference_rays_o + reference_depths[:,None] * reference_rays_d

    projected_points, projected_depths = perspective_projection_np(reference_points,
            target_intrinsic,
            target_pose, 
        )

    projected_points = projected_points.reshape(H,W,2)
    projected_depths = projected_depths.reshape(H,W)

    pred_projection_image, valid_mask = project_points_to_image( projected_points, projected_depths, reference_im, target_depths )
    return pred_projection_image, projected_depths, valid_mask

def multi_view_consistency( images, rays, poses, depths, H, W, K ):
    print('Performing multi-view consistency check of depth maps.')
    valid_depths = []
    #fig, axs = plt.subplots(10,2, figsize=(200,5))
    #fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0)
    for ref_idx in tqdm(range(len(images))):#len(images) #range(len(images))
        #ref_idx = 10
        reference_rays = rays[ref_idx]
        reference_depths = depths[ref_idx].copy()
        reference_im = images[ref_idx]
        t_idxs = [ref_idx-1,ref_idx+1,
                    ref_idx + len(images) // 2, 
                    ref_idx + len(images) // 2 + 1, ref_idx + len(images) // 2 - 1 ]
        valid_mask = np.ones((H,W)).astype(np.bool)
        for t_idx in t_idxs:
            if 0 <= t_idx < len(images):
                target_pose = poses[t_idx]
                target_intrinsic = K
                target_depths = depths[t_idx]
                pred_image, pred_depths, t_valid_mask = project_im(reference_rays, 
                                                        reference_depths, 
                                                        reference_im, 
                                                        target_pose, 
                                                        target_intrinsic, 
                                                        target_depths, H, W)
                #target_image = images[t_idx]
                if np.sum(t_valid_mask) > 0:
                    valid_mask = np.logical_and(valid_mask, t_valid_mask)

        #valid_reference_depths = reference_im.copy()
        #valid_reference_depths[np.logical_not(valid_mask),:] = np.array([0,0,0])
        valid_depth = depths[ref_idx].copy()
        valid_depth[np.logical_not(valid_mask)] = 0.
        valid_depths.append(valid_depth)


        '''axs[ref_idx,0].imshow(depths[ref_idx])
        axs[ref_idx,0].axis('off')
        axs[ref_idx,1].imshow(valid_depth)#reference_depths.reshape(H,W)
        axs[ref_idx,1].axis('off')'''
        
    return np.array(valid_depths)
    '''fig, axs = plt.subplots(2,2, figsize=(20,5))
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0)
    axs[0,0].imshow(reference_im)
    axs[0,0].axis('off')
    axs[0,1].imshow(target_image)#reference_depths.reshape(H,W)
    axs[0,1].axis('off')

    #axs[1,0].imshow(images[projection_idx])
    #axs[1,0].axis('off')
    axs[1,0].imshow(pred_image[:,:,:3])
    axs[1,0].axis('off')
    axs[1,1].imshow(valid_reference_depths)#valid_mask,cmap='gray'
    axs[1,1].axis('off')'''

    #plt.savefig('out.png')
    #sys.exit(0)

if __name__ == "__main__":

    data_root = '/BS/databases16/KITTI-360/'
    dropout= 0.8
    block = 0
    images, poses, depths_gt, test_poses, render_poses, random_poses, hwf, near, far, _ = load_kitti(
            data_root = data_root, 
            mode = 'test', 
            dropout = dropout,
            block = block, 
            coord_scale_factor=50.0,
            so3_representation = 'quaternion', 
            intrinsics_representation = 'square',
            initial_fov = 53.13,
            reg_pose_list = None,
            lr_check = False
    )
    H, W, focal = hwf
    H, W = int(H), int(W)
    K = np.array([
                [focal, 0, 682.049453],
                [0, focal, 238.769549],
                [0, 0, 1]
        ])
    rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0)
    valid_depths = multi_view_consistency( images, rays, poses, depths_gt, H, W, K )

    _,train_frames,eval_frames = get_test_block_final(data_root,dropout,block)
    STDROOTL = os.path.join(data_root,im_data_loc,'2013_05_28_drive_' + str(drive_seq).rjust(4,'0') + '_sync', 
                        'image_00','SMD-results-left-mvc')#-left-lrc
    STDROOTR = os.path.join(data_root,im_data_loc,'2013_05_28_drive_' + str(drive_seq).rjust(4,'0') + '_sync', 
                        'image_00','SMD-results-right-mvc')#-lrc

    os.makedirs(STDROOTL, exist_ok=False)
    os.makedirs(STDROOTR, exist_ok=False)

    for index in tqdm(range(frame_range[1] - frame_range[0] + 1)):
        img_num_curr = frame_range[0] + index#ego_img_nums[ frame_range[0] + index ]
        #print('img_num_curr ',img_num_curr)
        if img_num_curr in train_frames:
            np.save( os.path.join(STDROOTL, str(int(img_num_curr)).rjust(10,'0') + '_disp.npy'), valid_depths[index] )
            

    for index in tqdm(range(frame_range[1] - frame_range[0] + 1)):
        img_num_curr = frame_range[0] + index#ego_img_nums[ frame_range[0] + index ]
        #print('img_num_curr ',img_num_curr)
        if img_num_curr in train_frames:
            np.save( os.path.join(STDROOTR, str(int(img_num_curr)).rjust(10,'0') + '_disp.npy'), valid_depths[index + (frame_range[1] - frame_range[0] + 1)] )
            

