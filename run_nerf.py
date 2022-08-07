import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
from matplotlib import cm

from run_nerf_helpers import *

from load_kitti import load_kitti, multi_view_consistency
from smdnet_utils import color_depth_map, color_error_image_kitti


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False
scene_bbox = torch.tensor([[ -34.3491, -8.15422, -40.5304 ], [ 38.5536, 30.0, 111.638 ]]).cuda()
box_center = ((scene_bbox[0] + scene_bbox[1]) / 2)
box_size = torch.abs(scene_bbox[1] - scene_bbox[0])


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def renormalize_to_unit_box(rays_pts):
    rays_pts = (rays_pts - box_center[None,:].cuda()) / (box_size[None,:].cuda() / 2)
    return rays_pts


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    inputs_flat = renormalize_to_unit_box(inputs_flat)

    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, gt_depths=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []
    depths = []
    depth_errors = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, depth, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        depths.append(depth.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename_rgb = os.path.join(savedir, '{:03d}.rgb.png'.format(i))
            imageio.imwrite(filename_rgb, rgb8)

            color_depth = color_depth_map(depths[-1], scale=0.8)
            filename_depth = os.path.join(savedir, '{:03d}.depth.png'.format(i))
            imageio.imwrite(filename_depth, color_depth)

        if gt_depths is not None:

            color_depth_gt = color_depth_map(gt_depths[i].copy(), scale=0.8)
            filename_depthgt = os.path.join(savedir, '{:03d}.depthgt.png'.format(i))
            imageio.imwrite(filename_depthgt, color_depth_gt)

            depth_error = np.abs((gt_depths[i].copy() - depths[i])*50.)
            depth_error = color_error_image_kitti(
                depth_error, 
                scale=1, 
                mask=((gt_depths[i] > 0.0) & (gt_depths[i] < 1.0)), 
                BGR=False, 
                dilation=1)
            #depth_error = (depth_error*255).astype(np.uint8)
            filename_deptherror = os.path.join(savedir, '{:03d}.deptherror.png'.format(i))
            imageio.imwrite(filename_deptherror, depth_error)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, 
                 W_density=args.netwidth_density,
                 W_viewdir=args.netwidth_viewdir,
                 input_ch=input_ch, 
                 output_ch=output_ch, 
                 skips=skips,
                 input_ch_views=input_ch_views,
                 density_activation=args.density_activation,
                 color_activation=args.color_activation, 
                 use_viewdirs=args.use_viewdirs).to(device)

    model = nn.DataParallel(model).cuda()
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, 
                          W_density=args.netwidth_density,
                          W_viewdir=args.netwidth_viewdir,
                          input_ch=input_ch, 
                          output_ch=output_ch, 
                          skips=skips,
                          density_activation=args.density_activation,
                          color_activation=args.color_activation, 
                          input_ch_views=input_ch_views, 
                          use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    exptype = args.config.split('/')[-2]
    expname = args.config.split('/')[-1][:-4]

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, exptype, expname, f) for f in sorted(os.listdir(os.path.join(basedir, exptype, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    render_kwargs_random = render_kwargs_train.copy()
    #render_kwargs_random['prob_volume_sampling'] = False
    #render_kwargs_random['prob_volume_s'] = None

    return render_kwargs_train, render_kwargs_test, render_kwargs_random, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    #raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    raw2alpha = lambda raw, dists, act_fn=F.softplus: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, depth_map_0 = rgb_map, disp_map, acc_map, depth_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['depth0'] = depth_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def get_random_patches( random_ray_cache, num_random_poses, novel_view_batch_size, novel_view_reg_patch_size, H, W):
    # Pick random patch coordinate
    patch_min_y = torch.randint(low=0, high=H - novel_view_reg_patch_size, size=(novel_view_batch_size,))
    patch_min_x = torch.randint(low=0, high=W - novel_view_reg_patch_size, size=(novel_view_batch_size,))
    patch_max_y = patch_min_y + novel_view_reg_patch_size
    patch_max_x = patch_min_x + novel_view_reg_patch_size

    # Build ray batch from precomputed ray cache
    batch_rays = []
    random_pose_indices = np.random.choice(num_random_poses, size=novel_view_batch_size)
    for random_pose_idx, min_y, max_y, min_x, max_x in zip(random_pose_indices, patch_min_y, patch_max_y, patch_min_x, patch_max_x):
        batch_rays.append(random_ray_cache[random_pose_idx, :, min_y:max_y, min_x:max_x].reshape(2, -1, 3))

    batch_rays = torch.stack(batch_rays, dim=1).view(2, -1, 3).to(device)
    return batch_rays

def compute_depth_tv_norm_loss(args, expected_depth, acc):
    acc = acc.view(args.novel_view_batch_size,
        args.novel_view_reg_patch_size, args.novel_view_reg_patch_size)[:, :-1, :-1]
    if args.depth_tvnorm_detach_random_acc:
        acc = acc.detach()
    expected_depth = expected_depth.view(args.novel_view_batch_size,
        args.novel_view_reg_patch_size, args.novel_view_reg_patch_size)

    return  compute_tv_norm(expected_depth, args.depth_tvnorm_type, acc * args.depth_tvnorm_mask_weight).mean()


def plot_rays_to_patches( args, random_rgb ):
    #print('random_rgb ',random_rgb.shape)
    random_rgb = random_rgb.view(args.novel_view_batch_size,
        args.novel_view_reg_patch_size, args.novel_view_reg_patch_size,3)

    random_rgb[torch.isnan(random_rgb)] = 0.
    random_rgb = torch.clamp(random_rgb, 0., 1.)
    #print('random_rgb ',random_rgb.shape)
    random_rgb = random_rgb.permute(0,3,1,2)
    vutils.save_image(random_rgb.clone().detach().cpu(), './out.png', normalize=True)


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/kitti', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth_density", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netwidth_viewdir", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--density_activation", type=str, default='relu', 
                        help='MLP density activation')
    parser.add_argument("--color_activation", type=str, default='relu', 
                        help='MLP density activation')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    

    # org nerf decay
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')

    # mip nerf decay

    parser.add_argument("--mip_nerf_decay", action='store_true', help='use mip nerf decay')
    parser.add_argument("--lr_init", type=float, default=5e-4,  help='init learning rate')
    parser.add_argument("--lr_final", type=float, default=5e-6, help='final learning rate')
    parser.add_argument("--lr_delay_steps", type=float, default=2500,  help='init learning rate')
    parser.add_argument("--lr_delay_mult", type=float, default=0.01, help='final learning rate')


    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # depth supervision
    parser.add_argument("--depth_loss_weight", type=float, default=0., help='depth loss weight')
    parser.add_argument("--depth_coarse_loss_weight", type=float, default=0., help='depth coarse loss weight')

    # Depth smoothness loss
    parser.add_argument('--novel_view_reg_patch_size', type=int, default=8)
    parser.add_argument('--novel_view_batch_size', type=int, default=16) # RegNeRF uses 64 (resulting in 4096 rays)

    parser.add_argument('--depth_tvnorm_loss_mult', type=float, default=0.)
    parser.add_argument('--depth_tvnorm_decay', action='store_true')
    parser.add_argument('--depth_tvnorm_maxstep', type=int, default=0)
    parser.add_argument('--depth_tvnorm_loss_mult_start', type=float, default=0.)
    parser.add_argument('--depth_tvnorm_loss_mult_end', type=float, default=0.)
    parser.add_argument('--depth_tvnorm_mask_weight', type=float, default=0.)
    parser.add_argument('--depth_tvnorm_type', type=str, default='l2')

    parser.add_argument('--depth_tvnorm_detach_random_acc', action='store_true')

    parser.add_argument('--depth_tvnorm_coarse_loss_mult', type=float, default=0.)
    parser.add_argument('--depth_tvnorm_coarse_decay', action='store_true')
    parser.add_argument('--depth_tvnorm_coarse_maxstep', type=int, default=0)
    parser.add_argument('--depth_tvnorm_coarse_loss_mult_start', type=float, default=0.)
    parser.add_argument('--depth_tvnorm_coarse_loss_mult_end', type=float, default=0.)


    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=2, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## kitti360 flags
    parser.add_argument("--kitti_split", type=str, default='test', help='leaderboard test block')
    parser.add_argument("--block", type=int, default=0, help='leaderboard test block')
    parser.add_argument("--coord_scale_factor", type=float, default=50., help='leaderboard test block')
    parser.add_argument("--multi_view_consistency", action='store_true', 
                        help='perform multiview consistency check of depth maps')

    parser.add_argument("--reg_pose_list", type=str, help='list of poses for depth regularization')
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    parser.add_argument("--i_traindepth",   type=int, default=50000, 
                        help='frequency of evaluating depth fit.')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None

    if args.dataset_type == 'kitti360':
        images, poses, depths_gt, test_poses, render_poses, random_poses, hwf, near, far, _ = load_kitti(
                                                        data_root = args.datadir, 
                                                        mode = args.kitti_split, 
                                                        dropout= 0.8 if args.kitti_split == 'test' else None,
                                                        block = args.block, 
                                                        coord_scale_factor=50.0,
                                                        so3_representation = 'quaternion', 
                                                        intrinsics_representation = 'square',
                                                        initial_fov = 53.13,
                                                        reg_pose_list = args.reg_pose_list)

        i_train = list(range(len(images)))
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        if args.dataset_type == 'kitti360':
            K = np.array([
                [focal, 0, 682.049453],
                [0, focal, 238.769549],
                [0, 0, 1]
            ])
        else:
            pass

    
    if args.multi_view_consistency:
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0)
        depths_gt = multi_view_consistency( images, rays, poses, depths_gt, H, W, K )
    #if args.render_test:
    #    render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    exptype = args.config.split('/')[-2]
    os.makedirs(os.path.join(basedir, exptype), exist_ok=True)
    expname = args.config.split('/')[-1][:-4]
    os.makedirs(os.path.join(basedir, exptype, expname), exist_ok=True)
    f = os.path.join(basedir, exptype, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, exptype, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, render_kwargs_random, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    render_kwargs_random.update(bds_dict)

    # Move pose data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    test_poses = torch.Tensor(test_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            '''if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None'''

            testsavedir = os.path.join(basedir, exptype, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'train', start))
            os.makedirs(testsavedir, exist_ok=True)

            rgbs, _ = render_path(torch.Tensor(test_poses).to(device) if args.render_test else torch.Tensor(poses).to(device), 
                hwf, K, args.chunk, render_kwargs_test, 
                gt_imgs=None, 
                gt_depths=None if args.render_test else depths_gt, 
                savedir=testsavedir, 
                render_factor=args.render_factor
            )
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        tmp_depths = np.repeat(depths_gt[:,None,:,:,None], 3, axis=-1)
        rays_rgbd = np.concatenate([rays, images[:,None], tmp_depths], 1) # [N, ro+rd+rgb+depth, H, W, 3]
        rays_rgbd = np.transpose(rays_rgbd, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb+depth, 3]
        rays_rgbd = np.reshape(rays_rgbd, [-1,4,3]) # [(N-1)*H*W, ro+rd+rgb+depth, 3]
        rays_rgbd = rays_rgbd.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgbd)

        print('done')
        i_batch = 0

    
    # Move training data to GPU
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        images = torch.Tensor(images).to(device)
        rays_rgbd = torch.Tensor(rays_rgbd).to(device)

    if args.depth_tvnorm_decay or args.depth_tvnorm_coarse_decay:
        random_ray_cache = torch.empty(len(random_poses), 2, H, W, 3, device=torch.device('cpu'))
        for random_pose_idx in range(len(random_poses)):
            random_pose = random_poses[random_pose_idx, :3, :4]
            rays_o, rays_d = get_rays(H, W, K, torch.Tensor(random_pose))  # (H, W, 3), (H, W, 3)
            random_ray_cache[random_pose_idx] = torch.stack([rays_o, rays_d], 0)


    N_iters = 2000000 + 1
    print('Begin')
    #print('TRAIN views are', i_train)
    #print('TEST views are', i_test)
    #print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    depth_loss = None
    for i in trange(start, N_iters):
        time0 = time.time()

        if args.depth_tvnorm_decay:
            depth_tvnorm_loss_weight = compute_tvnorm_weight(i, args.depth_tvnorm_maxstep,
                args.depth_tvnorm_loss_mult_start, args.depth_tvnorm_loss_mult_end)
        else:
            depth_tvnorm_loss_weight = args.depth_tvnorm_loss_mult

        if args.depth_tvnorm_coarse_decay:
            depth_tvnorm_coarse_loss_weight = compute_tvnorm_weight(i, args.depth_tvnorm_coarse_maxstep,
                args.depth_tvnorm_coarse_loss_mult_start, args.depth_tvnorm_coarse_loss_mult_end)
        else:
            depth_tvnorm_coarse_loss_weight = args.depth_tvnorm_coarse_loss_mult

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgbd[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s, target_d = batch[:2], batch[2], batch[3]
            target_d = target_d[:,0]

            i_batch += N_rand
            if i_batch >= rays_rgbd.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgbd.shape[0])
                rays_rgbd = rays_rgbd[rand_idx]
                i_batch = 0

        '''else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)'''

        #####  Core optimization loop  #####
        rgb, disp, acc, depth, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        #print('extras ',extras.keys())
        #sys.exit(0)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        
        trans = extras['raw'][...,-1]


        if args.depth_coarse_loss_weight > 0:
            depth_loss_coarse = maskedimg2mse(extras['depth0'], target_d, (target_d > 0) & (target_d < 1.) )
            loss = loss + args.depth_coarse_loss_weight*depth_loss_coarse

        if args.depth_loss_weight > 0:
            depth_loss = maskedimg2mse(depth, target_d, (target_d > 0) & (target_d < 1.) )
            loss = loss + args.depth_loss_weight*depth_loss

        if args.depth_tvnorm_decay or args.depth_tvnorm_coarse_decay:
            random_rays = get_random_patches( random_ray_cache, len(random_poses), 
                args.novel_view_batch_size, args.novel_view_reg_patch_size, H, W)

            #print('random_rays ',random_rays.shape)

            random_rgb, _, random_acc, random_depth, random_extras = render(H, W, K, chunk=args.chunk, rays=random_rays,
                                                    verbose=i < 10, retraw=True,
                                                    **render_kwargs_random)

            #print('random_acc, random_depth ',random_acc.shape, random_depth.shape)

            if i % 100 == 0:
                plot_rays_to_patches( args, random_rgb )

            if depth_tvnorm_loss_weight > 0.:
                depth_tv_norm_loss = depth_tvnorm_loss_weight * compute_depth_tv_norm_loss(args, random_depth, random_acc)
                loss = loss + depth_tv_norm_loss

            if depth_tvnorm_coarse_loss_weight > 0.:
                depth_tv_norm_coarse_loss = depth_tvnorm_coarse_loss_weight * compute_depth_tv_norm_loss(args, random_extras['depth0'], random_extras['acc0'])
                loss = loss + depth_tv_norm_coarse_loss

        
        psnr = mse2psnr(img_loss)

        '''if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)'''

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###

        #if not args.mip_nerf_decay:
        if global_step > 4000:
            decay_rate = 0.96
            decay_steps = args.lrate_decay #* 1000
            new_lrate = args.lrate * (decay_rate ** int((global_step - 4000) // decay_steps))
        else:
            warmup_alpha = global_step / 4000.
            warmup_alpha = 1. - warmup_alpha
            new_lrate = 1e-5 * warmup_alpha + (1. - warmup_alpha) * args.lrate
        '''else:
            new_lrate = learning_rate_decay(global_step,
                        lr_init = args.lr_init,
                        lr_final = args.lr_final,
                        max_steps = N_iters,
                        lr_delay_steps=args.lr_delay_steps,
                        lr_delay_mult=args.lr_delay_mult)'''


        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################




        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, exptype, expname, '{:06d}.tar'.format(i))
            if args.N_importance > 0:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, depths = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, depths.shape)
            moviebase = os.path.join(basedir, exptype, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            #imageio.mimwrite(moviebase + 'depths.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, exptype, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', test_poses.shape)
            with torch.no_grad():
                render_path(test_poses, hwf, K, args.chunk, 
                render_kwargs_test, gt_imgs=None, savedir=testsavedir)#images[i_test]
            print('Saved test set')


    
        if i%args.i_print==0:
            if depth_loss is not None:
                print(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} Depth Loss: {depth_loss.item()}")
            else:
                print(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
