import numpy as np
import matplotlib.pyplot as plt


COLORS_PLT = [(1, 0, 0),
              (0, 1, 0),
              (0, 0, 1),
              (0, 1, 1),
              (1, 0, 1),
              (1, 1, 0),
              (0.5, 0, 0.9),
              (0.9, 0, 0.5)]


def get_rotation_matrix(axis, angle):

    angle = (angle/360)*2*np.pi

    if axis == 0:
        return np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    if axis == 1:
        return np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    if axis == 2:
        return np.array([[np.cos(angle),  -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])


def rotate_np(pc, angle_axis0, angle_axis1, angle_axis2):
    mat1 = get_rotation_matrix(axis=0, angle=angle_axis0)
    mat2 = get_rotation_matrix(axis=1, angle=angle_axis1)
    mat3 = get_rotation_matrix(axis=2, angle=angle_axis2)
    rot_mat = mat1.dot(mat2).dot(mat3)
    return np.einsum('ij,kjl->kil', rot_mat, pc)


def add_figures_reconstruction_tb(imgs_gt, imgs_reconst, mixture_labels, summary_writer, iter, nr_samples: int = 5):

    imgs_gt = rotate_np(imgs_gt, 25, 135, 0)
    imgs_reconst = rotate_np(imgs_reconst, 25, 135, 0)
    fig, axs = plt.subplots(nr_samples, 2, figsize=(15, 15))
    for i in range(nr_samples):
        axs[i, 0].scatter(imgs_gt[i, 0, :], imgs_gt[i, 1, :], s=10.0, alpha=0.5)
        c = [COLORS_PLT[lbl - 1] for lbl in mixture_labels[i].astype(np.int)]
        axs[i, 1].scatter(imgs_reconst[i, 0, :], imgs_reconst[i, 1, :], s=10.0, alpha=0.5, c=c)
    summary_writer.add_figure('GT_vs_RECONSTRUCTION', fig, iter)

def add_svr_reconstruction_tb(imgs, imgs_gt, imgs_reconst, mixture_labels, summary_writer, iter, nr_samples: int = 5):
    imgs_gt = rotate_np(imgs_gt, 25, 135, 0)
    imgs_reconst = rotate_np(imgs_reconst, 25, 135, 0)
    fig, axs = plt.subplots(nr_samples, 3, figsize=(15, 15))
    for i in range(nr_samples):
        axs[i, 0].scatter(imgs_gt[i, 0, :], imgs_gt[i, 1, :], s=10.0, alpha=0.5)
        c = [COLORS_PLT[lbl - 1] for lbl in mixture_labels[i].astype(np.int)]
        axs[i, 1].scatter(imgs_reconst[i, 0, :], imgs_reconst[i, 1, :], s=10.0, alpha=0.5, c=c)
        axs[i, 2].imshow(imgs[i, :, :, 1:4])
    summary_writer.add_figure('GT_vs_RECONSTRUCTION', fig, iter)

