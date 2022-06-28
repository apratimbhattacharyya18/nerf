import sys
import os
import torch
import pickle
import glob
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
import pandas as pd

from pyntcloud import PyntCloud

from PIL import Image
import matplotlib.pyplot as plt

from nerf.run_nerf_helpers import *


with open('data/dtu_regnerf/train.txt') as f:
    train_scenes_raw = f.readlines()

train_scenes = []
for i in range(len(train_scenes_raw)):
    train_scenes.append(train_scenes_raw[i].strip())


def project_points_to_image( projection_points, projection_depths, reference_im ):
    projection_image = np.zeros((reference_im.shape[0],
        reference_im.shape[1],
        reference_im.shape[2]+1)
    )
    H, W = reference_im.shape[0], reference_im.shape[1]
    projection_image[:,:,3] = float('inf')
    projection_points = projection_points.astype(np.int32)
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
    return projection_image

#images, rays, poses, depths, masks

def project_im(reference_rays, reference_depths, target_pose, target_intrinsic, H, W):
    reference_rays_o, reference_rays_d = reference_rays[0],  reference_rays[1]
    reference_rays_o = reference_rays_o.reshape(-1,3)
    reference_rays_d = reference_rays_d.reshape(-1,3)

    reference_depths = reference_depths.reshape(-1)
    reference_points = reference_rays_o + reference_depths[:,None] * reference_rays_d

    projected_points, projected_depths = perspective_projection_dtu_np(reference_points,
            target_intrinsic,
            target_pose, 
            )

    projected_points = projected_points.reshape(H,W,2)
    projected_depths = projected_depths.reshape(H,W)

    pred_projection_image = project_im( projected_points, projected_depths, images[reference_idx] )

def multi_view_consistency( images, rays, poses, depths, masks, Ks ):
    H, W = images[0].shape[:2]
    reference_idx = len(images) // 2 + 1
    reference_rays = rays[reference_idx]
    
    reference_depths = depths[reference_idx]
    

    projection_idx = reference_idx - 1

    Ks[projection_idx], poses[projection_idx]

    #print('projection_points ',projection_points.shape)
    

    projection_depths = depths[projection_idx]

    occ_mask = ~((~(masks[projection_idx] > 0.)) | (np.abs(projected_depths - projection_depths) > 60))

    
    fig, axs = plt.subplots(2,2)
    axs[0,0].imshow(images[reference_idx])
    axs[0,0].axis('off')
    axs[0,1].imshow(images[projection_idx])#reference_depths.reshape(H,W)
    axs[0,1].axis('off')

    #axs[1,0].imshow(images[projection_idx])
    #axs[1,0].axis('off')
    axs[1,0].imshow(pred_projection_image[:,:,:3])
    axs[1,0].axis('off')
    axs[1,1].imshow(occ_mask,cmap='gray')
    axs[1,1].axis('off')

    plt.savefig('out.png')
    sys.exit(0)


def load_dtu_data(basedir, scan, half_res=False, testskip=1, no_val=False):

    if True:
        with open('/BS/databases16/DTU_extn/dump.pkl', 'rb') as f:
            data = pickle.load(f)

        return data['images'], data['poses'], data['depths'], data['masks'], data['Ks'], data['hwf'], data['near'], data['far'], data['camtoworlds'], data['im_scene_ids'], None


    print('Loading ... ',basedir)#scan
    im_dir = os.path.join(basedir, 'Rectified_rescaled','0.25')#,'scan30'
    #depth_dir = os.path.join(basedir, 'Depth','0.25')#,'scan30'
    mask_dir = os.path.join(basedir, 'planed_Depth','0.25')#,'scan30'
    pose_dir = os.path.join(basedir, 'Calibration','cal18')

    im_files = glob.glob(im_dir + '/*/rect_*_max.png')#*/
    #depth_files = glob.glob(depth_dir + '/*/rect_*_points.npy')#*/
    #mask_files = glob.glob(mask_dir + '/*/rect_*_points.png')#*/
    #pose_files = glob.glob(pose_dir + '/pos_*.txt')

    

    im_files.sort(key = lambda x: (int(x.split('/')[-2][4:]), int(x.split('/')[-1].split('_')[1])))
    #depth_files.sort(key = lambda x: (int(x.split('/')[-2][4:]), int(x.split('/')[-1].split('_')[1])))
    #mask_files.sort(key = lambda x: (int(x.split('/')[-2][4:]), int(x.split('/')[-1].split('_')[1])))
    #pose_files.sort(key = lambda x: int(x.split('/')[-1].split('_')[1][:-4]))

    images, depths, masks, camtoworlds, pixtocams, Ks, im_scene_ids  = [], [], [], [], [], [], []

    im_files = im_files#[:5*49]#[2*48+2:3*48]
    #depth_files = depth_files#[:10*49]#[2*48+2:3*48]
    #mask_files = mask_files#[:10*49]#[2*48+2:3*48]
    #pose_files = pose_files[2*48+2:3*48]

    curr_scene_idx = 0
    prev_scene_name = im_files[0].split('/')[-2]

    for im_file in im_files:

        #if not ('scan114/' in im_file):
        #    continue

        curr_scene_name = im_file.split('/')[-2]
        #print('curr_scene_name ',curr_scene_name)

        if curr_scene_name not in train_scenes:
            continue

        if curr_scene_name != prev_scene_name:
            prev_scene_name = curr_scene_name
            curr_scene_idx += 1
        
        im_file_num = int(im_file.split('/')[-1].split('_')[1])
        extn_num = str(im_file_num - 1).rjust(3,'0')
        print('extn_num ',extn_num)

        depth_file = im_file.replace('Rectified_rescaled','Depth')
        depth_file = depth_file[:-11] + extn_num + '_points.npy'

        mask_file = im_file.replace('Rectified_rescaled','planed_Depth')
        mask_file = mask_file[:-11] + extn_num + '_points.png'

        print('im_file, depth_file, mask_file ',im_file, depth_file, mask_file)

        if not os.path.exists(depth_file) or not os.path.exists(mask_file):
            continue

        im = np.array(Image.open(im_file).convert('RGB'))/255.
        depth = np.load(depth_file)
        mask = np.array(Image.open(mask_file).convert('L'))/255.
        #mask = np.mean(im,axis=-1)
        #print('mask ',np.amin(mask),np.amax(mask))
        #mask = 1 - mask
        mask = (mask > 0.).astype(np.float32)#((mask > 1e-2) & (mask < (1. - 1e-2))).astype(np.float32)
        '''fig, axs = plt.subplots(1,2)
        axs[0].imshow(im)
        axs[0].axis('off')
        axs[1].imshow(mask)
        axs[1].axis('off')
        plt.savefig('out.png')
        sys.exit(0)'''
        images.append(im)
        masks.append(mask)
        depths.append(depth)
        im_scene_ids.append(curr_scene_idx)
        pose_file = os.path.join(pose_dir, 'pos_' + im_file.split('/')[-1].split('_')[1] + '.txt')

        with open(pose_file, 'rb') as f:
            projection = np.loadtxt(f, dtype=np.float32)

        # Decompose projection matrix into pose and camera matrix.
        camera_mat, rot_mat, t = cv2.decomposeProjectionMatrix(projection)[:3]
        camera_mat = camera_mat / camera_mat[2, 2]
        Ks.append(camera_mat)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot_mat.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]
        #pose = pose[:3]
        camtoworlds.append(pose.copy())
        pixtocams.append(np.linalg.inv(camera_mat))

        #if len(camtoworlds) >= 49*5:
        #    break


        '''if len(camtoworlds) >= 3:
            H, W = images[0].shape[:2]
            images = np.array(images)
            masks = np.array(masks)
            fig, axs = plt.subplots(1,6)
            axs[0].imshow(images[0])
            axs[1].imshow(images[1])
            axs[2].imshow(images[2])

            axs[0].axis('off')
            axs[1].axis('off')
            axs[2].axis('off')

            #axs[3].imshow(images[3])
            #axs[4].imshow(images[4])
            #axs[5].imshow(images[5])

            #axs[3].axis('off')
            #axs[4].axis('off')
            #axs[5].axis('off')

            plt.savefig('out_ims.png')
            poses = np.array(camtoworlds)
            factor = 4.
            for idx in range(len(Ks)):
                camera_mat = Ks[idx]
                camera_mat = np.diag(
                [1./factor,
                 1./factor, 1.]).astype(np.float32) @ camera_mat
                Ks[idx] = camera_mat
            rays = np.stack([get_rays_np_dtu(H, W, K, p) for p, K in zip(poses[:,:3,:4],Ks)], 0)
            depths = np.array(depths)
            print('rays, depths ',rays.shape, images.shape, depths.shape)

            multi_view_consistency( images, rays, poses, depths, masks, Ks )

            all_points = []
            all_colors = []
            for idx in range(3):
                curr_rays = rays[idx]
                rays_o, rays_d = curr_rays[0],  curr_rays[1]
                rays_o = rays_o.reshape(-1,3)
                rays_d = rays_d.reshape(-1,3)
                reference_depths = depths[idx]
                reference_depths = reference_depths.reshape(-1)
                points = rays_o + reference_depths[:,None] * rays_d
                all_points.append(points)
                color = images[idx].reshape(-1,3)*255.#np.zeros_like(points)
                #color[:,idx] = 255.
                all_colors.append(color)

            all_points = np.concatenate(all_points, axis=0)
            all_colors = np.concatenate(all_colors, axis=0).astype(np.uint8)
            print('all_points, all_colors ',all_points.shape, all_colors.shape)

            d = {'x': all_points[:,0],'y': all_points[:,1],'z': all_points[:,2], 
             'red' : all_colors[:,0], 'green' : all_colors[:,1], 'blue' : all_colors[:,2]}

            cloud = PyntCloud(pd.DataFrame(data=d))
            cloud.to_file("output.ply")
            print('output.ply')
            sys.exit(0)'''

        
    #print('images[0].shape ',images[0].shape)
    H, W = images[0].shape[:2]
    hwf = [H,W,float(camera_mat[0,0])]
    near = 0.5
    far = 3.5

    factor = 4.
    for idx in range(len(Ks)):
        camera_mat = Ks[idx]
        camera_mat = np.diag(
        [1./factor,
         1./factor, 1.]).astype(np.float32) @ camera_mat
        Ks[idx] = camera_mat

    images = np.array(images)
    poses = np.array(camtoworlds)
    depths = np.array(depths)
    masks = np.array(masks)
    im_scene_ids = np.array(im_scene_ids)

    '''dict_to_save = {'images':images,'poses':poses,'depths':depths,'masks':masks,
                'Ks':Ks,'hwf':hwf, 'near':near, 'far':far, 'camtoworlds':camtoworlds, 
                'im_scene_ids':im_scene_ids}

    with open('/BS/databases16/DTU_extn/dump.pkl', 'wb') as f:
        pickle.dump(dict_to_save, f, protocol=4)
    #sys.exit(0)'''

    return images, poses, depths, masks, Ks, hwf, near, far, camtoworlds, im_scene_ids, None