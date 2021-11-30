import os
import argparse
import torch
import numpy as np
import h5py
import open3d
from multiprocessing import Process

from lib.metrics.evaluation_metrics import distChamfer
from lib.visualization.utils_open3d import numpy2ply


def define_options_parser():
    parser = argparse.ArgumentParser(description='Model training script. Provide a suitable config.')
    parser.add_argument('experiment_path', type=str, help='Path to experiment which contains .npy-files.')
    parser.add_argument('nr_samples', type=int, help='Number figures.')
    return parser


def vis_process(samples, labels, heatmap):
    for i in range(0, args.nr_samples):
        smp_cur = samples[i]
        if labels is not None:
            labels_cur = labels[i]
        else:
            labels_cur = None
        samples_point_cloud = numpy2ply(smp_cur, labels_cur, heatmap=heatmap)
        open3d.visualization.draw_geometries([samples_point_cloud], width=640, height=480)


if __name__ == '__main__':

    parser = define_options_parser()
    args = parser.parse_args()

    smp = np.load(os.path.join(args.experiment_path, 'all_samples.npy'))
    gts = np.load(os.path.join(args.experiment_path, 'all_gts.npy'))
    smp = smp[20:args.nr_samples+20]
    gts = gts[20:args.nr_samples+20]
    chamfer_distance = distChamfer(torch.from_numpy(smp).transpose(2, 1), torch.from_numpy(gts).transpose(2, 1))
    labels = np.load(os.path.join(args.experiment_path, 'all_labels.npy'))
    labels = labels[20:args.nr_samples+20]

    # view gt, sample and heatmap of prediction error in parallel.
    p1 = Process(target=vis_process, args=(gts, None, False))
    p1.start()
    p2 = Process(target=vis_process, args=(smp, labels, False))
    p2.start()
    p3 = Process(target=vis_process, args=(smp, chamfer_distance[1], True))
    p3.start()
    p1.join()
    p2.join()
    p3.join()
