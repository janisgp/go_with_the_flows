import h5py
import torch
#from lib.metrics.evaluation_metrics import  _pairwise_EMD_CD_F1_SCORE, distChamfer
from visualize_flow_mixture_output import single_point_cloud_render
import sys
sys.path.append('.')
import os

import matplotlib.pyplot as plt
from lib.visualization.utils import rotate_np, COLORS_PLT
from lib.visualization.utils_open3d import numpy2ply
import open3d
import numpy as np
import argparse
from plyfile import PlyData, PlyElement
import cv2

def define_options_parser():
    parser = argparse.ArgumentParser(description='Model training script. Provide a suitable config.')
    parser.add_argument('experiment_path', type=str, help='Path to experiment which contains .npy-files.')
    parser.add_argument('nr_samples', type=int, help='Number figures.')
    parser.add_argument('h5_file', type=str, help='generated h5 file containing gt, samples and labels.')
    parser.add_argument('mode', type=str, choices=['generation', 'reconstruction', 'autoencoding'],
                        help='choose one mode of h5 file')
    parser.add_argument('visualization', type=str, choices=['open3d', 'mitsuba', 'matplotlib'],
                        help='choose one mode to see the visualization')
    parser.add_argument('--plot_cd', action='store_true',
                        help='flag to plot cd distribution or not')
    return parser

def DeNormalizeImages(image):
        image_means = [0.03492457, 0.03379815, 0.03475684, 0.03874264]
        image_stds = [0.10963749, 0.10795733, 0.11031612, 0.12266339]
        mean = np.array(image_means, dtype=np.float32)
        std = np.array(image_stds, dtype=np.float32)
        return (image * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1))

def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]
if __name__=='__main__':
    parser = define_options_parser()
    args = parser.parse_args()
    f = h5py.File(os.path.join(args.experiment_path, args.h5_file), "a")
    gt_clouds = torch.from_numpy(f['gt_clouds'][:])
    sampled_clouds = torch.from_numpy(f['sampled_clouds'][:])
    if args.mode == 'reconstruction':
        images = f['image_clouds'][:]
    print("data loaded.")

    gt_clouds = torch.transpose(gt_clouds, 1, 2)#.contiguous().cuda()
    sampled_clouds = torch.transpose(sampled_clouds, 1, 2)#.contiguous().cuda()      # N * npoints * 3
    if args.mode == 'generation':
        M_rs_cd, M_rs_emd, _, _, _ = _pairwise_EMD_CD_F1_SCORE(
            sampled_clouds, gt_clouds, batch_size=60, accelerated_cd=True,
            f1_threshold=0.0001, cd_option=True,
            one_part_of_cd=False, emd_option=False, f1_option=False)
        labels = f['sampled_labels'][:]
        N_sample, N_ref = M_rs_cd.size(0), M_rs_cd.size(1)
        val_fromsmp, idx = torch.min(M_rs_cd, dim=1)
        val, smp_idx = torch.min(M_rs_cd, dim=0)
        sampled_clouds = sampled_clouds[smp_idx, :, :]
        gt_clouds = gt_clouds[smp_idx,:, :]
        gt = gt_clouds.detach().cpu().numpy()
        smp = sampled_clouds[smp_idx, :, :].detach().cpu().numpy()
        labels = labels[smp_idx.cpu().numpy(), :]
        gt = gt.transpose(0, 2, 1)
        smp = smp.transpose(0, 2, 1)
    else:
        gt = gt_clouds.detach().cpu().numpy()
        smp = sampled_clouds.detach().cpu().numpy()
        gt = gt.transpose(0, 2, 1)
        smp = smp.transpose(0, 2, 1)
        labels = f['sampled_labels'][:]
    if args.visualization == 'open3d':
        for i in range(0, args.nr_samples):
            print(i)
            '''
            #see also gt
            gt_cur = gt[i * 24]
            labels_gt = [0 for _ in range(gt_cur.shape[-1])]
            samples_gt = numpy2ply(gt_cur, labels_gt)
            open3d.visualization.draw_geometries([samples_gt], width=640, height=480)
            '''
            smp_cur = smp[i*24]
            labels_cur = labels[i*24]
            samples_point_cloud = numpy2ply(smp_cur, labels_cur)
            open3d.visualization.draw_geometries([samples_point_cloud], width=640, height=480)

    elif args.visualization == 'mitsuba':
        k = [48]

        for i in range(0, len(k)):
            ## this part used to visualize multi flows
            smp_cur = smp[k[i] * 24, :, :5000]
            labels_cur = labels[k[i] * 24][:10000]
            #labels_cur = [4 for _ in range(5000)] #label = 4 used to visualize one flow model
            single_point_cloud_render(smp_cur, labels_cur, k[i], args.experiment_path)

            #save also image
            '''
            image_cur = images[k[i] * 24]
            image_cur = DeNormalizeImages(image_cur)
            image_cur = image_cur.transpose(1, 2, 0)
            plt.imshow((image_cur[:, :, :] * 255).astype(np.uint8))
            plt.axis('off')
            plt.savefig(os.path.join(args.experiment_path, 'rendered_pcl/%d.png')%k[i])
            plt.close()
            #visualize gt
            gt_cur = gt[k[i] * 24, :, :5000]
            labels_cur = [4 for _ in range(5000)]
            single_point_cloud_render(gt_cur, labels_cur, k[i], args.experiment_path)
            '''
    elif args.visualization == 'matplotlib':
        imgs_gt = rotate_np(gt, 25, 135, 0)
        imgs_reconst = rotate_np(smp, 25, 135, 0)
        n = 3 if args.mode == 'reconstruction' else 2
        fig, axs = plt.subplots(args.nr_samples, n, figsize=(15, 15))
        k = [128, 136, 183, 208, 228, 232, 303, 346, 351, 352, 378, 388]
        for i in range(0, args.nr_samples):
            #k = i * 24 if args.mode == 'reconstruction' else i
            axs[i, 0].scatter(imgs_gt[k[i], 0, :], imgs_gt[k[i], 1, :], s=10.0, alpha=0.5)
            c = [COLORS_PLT[lbl - 1] for lbl in labels[k[i]]]
            axs[i, 1].scatter(imgs_reconst[k[i], 0, :], imgs_reconst[k[i], 1, :], s=10.0, alpha=0.5, c=c)
            if args.mode == 'reconstruction':
                axs[i, 2].imshow(images[k[i], :, :, 1:4])
        plt.savefig(os.path.join(args.experiment_path, 'plt_smp.png'))

    if args.plot_cd == True:
        cdl, cdr = distChamfer(sampled_clouds[:args.nr_samples, :, :], gt_clouds[:args.nr_samples, :, :])
        cdl = cdl.cpu().numpy()
        cdr = cdr.cpu().numpy()
        print(cdl.shape)
        n = 5 if args.mode == 'reconstruction' else 4
        fig, axs = plt.subplots(args.nr_samples, n, figsize=(15, 15))
        imgs_gt = rotate_np(gt, 25, 135, 0)
        imgs_reconst = rotate_np(smp, 25, 135, 0)
        for i in range(0, args.nr_samples):
            axs[i, 0].scatter(imgs_gt[i, 0, :], imgs_gt[i, 1, :], s=10.0, alpha=0.5)
            c = [COLORS_PLT[lbl - 1] for lbl in labels[i].astype(np.int)]
            axs[i, 1].scatter(imgs_reconst[i, 0, :], imgs_reconst[i, 1, :], s=10.0, alpha=0.5, c=c)
            axs[i, 2].hist(cdl[i, :], bins=50)
            axs[i, 3].hist(cdr[i, :], bins=50)
            if args.mode == 'reconstruction':
                axs[i, 4].imshow(images[i, :, :, 1:4])
        plt.savefig(os.path.join(args.experiment_path, 'plt_cd.png'))