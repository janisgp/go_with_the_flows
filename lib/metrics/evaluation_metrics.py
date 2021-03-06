import torch
import numpy as np
import warnings
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Import CUDA version of approximate EMD, from https://github.com/zekunhao1995/pcgan-pytorch/
from .StructuralLosses.match_cost import match_cost
from .StructuralLosses.nn_distance import nn_distance


# # Import CUDA version of CD, borrowed from https://github.com/ThibaultGROUEIX/AtlasNet
# try:
#     from . chamfer_distance_ext.dist_chamfer import chamferDist
#     CD = chamferDist()
#     def distChamferCUDA(x,y):
#         return CD(x,y,gpu)
# except:

def distChamferCUDA(x, y):
    return nn_distance(x, y)


def emd_approx(sample, ref):
    B, N, N_ref = sample.size(0), sample.size(1), ref.size(1)
    assert N == N_ref, "Not sure what would EMD do in this case"
    emd = match_cost(sample, ref)  # (B,)
    emd_norm = emd / float(N)  # (B,)
    return emd_norm


# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
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

def EMD_CD_F1(sample_pcs, ref_pcs, batch_size, accelerated_cd=False, reduced=True, cd_option=False,
           emd_option=False, one_part_of_cd=False, f1_option=False, f1_threshold=0.0001):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    cd_lst = []
    emd_lst = []
    f1_lst = []
    cdl_lst = []
    cdr_lst= []
    cd, emd, f1_score, cdl, cdr = 0, 0, 0, 0, 0
    iterator = range(0, N_sample, batch_size)

    for b_start in iterator:
        b_end = min(N_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]

        if accelerated_cd:
            dl, dr = distChamferCUDA(sample_batch, ref_batch)
        else:
            dl, dr = distChamfer(sample_batch, ref_batch)
        if cd_option:
            cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))
        if one_part_of_cd:
            cdl_lst.append(dl.mean(dim=1))
            cdr_lst.append(dr.mean(dim=1))
        if emd_option:
            emd_batch = emd_approx(sample_batch, ref_batch)
            emd_lst.append(emd_batch)
        if f1_option:
            precision = 100. * (dr < f1_threshold).float().mean(1)
            recall = 100. * (dl < f1_threshold).float().mean(1)
            f1_score = 2. * precision * recall / (precision + recall + 1e-7)
            f1_lst.append(f1_score)

    if cd_option:
        cd = torch.cat(cd_lst).mean() if reduced else torch.cat(cd_lst)
    if emd_option:
        emd = torch.cat(emd_lst).mean() if reduced else torch.cat(emd_lst)
    if f1_option:
        f1_score = torch.cat(f1_lst).mean() if reduced else torch.cat(f1_lst)
    if one_part_of_cd:
        cdl = torch.cat(cdl_lst).mean() if reduced else torch.cat(cdl_lst)
        cdr = torch.cat(cdr_lst).mean() if reduced else torch.cat(cdr_lst)
        
    results = {
        'CD': cd,
        'EMD': emd,
        'F1': f1_score,
        'CDL': cdl,
        'CDR': cdr
    }
    return results


def _pairwise_EMD_CD_F1_SCORE(sample_pcs, ref_pcs, batch_size, f1_threshold, accelerated_cd=True,
                              cd_option=False, one_part_of_cd=False, emd_option=False, f1_option=False):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    all_f1_score = []
    all_cd_left = []
    all_cd_right = []
    iterator = range(N_sample)
    for sample_b_start in iterator:
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        f1_score_lst = []
        cd_lst_left = []
        cd_lst_right = []
        for ref_b_start in range(0, N_ref, batch_size):
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            if accelerated_cd:
                dl, dr = distChamferCUDA(sample_batch_exp, ref_batch)
            else:
                dl, dr = distChamfer(sample_batch_exp, ref_batch)

            if one_part_of_cd:
                cd_lst_left.append((dl.mean(dim=1)).view(1, -1))
                cd_lst_right.append((dr.mean(dim=1)).view(1, -1))

            if cd_option:
                cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))
                
            if emd_option:
                emd_batch = emd_approx(sample_batch_exp, ref_batch)
                emd_lst.append(emd_batch.view(1, -1))

            if f1_option:
                precision = 100. * (dr < f1_threshold).float().mean(1)
                recall = 100. * (dl < f1_threshold).float().mean(1)

                f1_score = 2. * precision * recall / (precision + recall + 1e-7)
                f1_score_lst.append(f1_score.view(1, -1))

        if cd_option:
            cd_lst = torch.cat(cd_lst, dim=1)
            all_cd.append(cd_lst)
        if emd_option:
            emd_lst = torch.cat(emd_lst, dim=1)
            all_emd.append(emd_lst)
        if f1_option:
            f1_score_lst = torch.cat(f1_score_lst, dim=1)
            all_f1_score.append(f1_score_lst)
        if one_part_of_cd:
            cd_lst_left = torch.cat(cd_lst_left, dim=1)
            cd_lst_right = torch.cat(cd_lst_right, dim=1)
            all_cd_left.append(cd_lst_left)
            all_cd_right.append(cd_lst_right)
    
    if cd_option:
        all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    if emd_option:
        all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref
    if f1_option:
        all_f1_score = torch.cat(all_f1_score, dim=0)
    if one_part_of_cd:
        all_cd_left = torch.cat(all_cd_left, dim=0)
        all_cd_right = torch.cat(all_cd_right, dim=0)
    return all_cd, all_emd, all_f1_score, all_cd_left, all_cd_right


# Adapted from https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s

def lgan_mmd_cov(all_dist, mode='min'):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    if mode == 'min':
        val_fromsmp, idx = torch.min(all_dist, dim=1)
        val, idx_mmd = torch.min(all_dist, dim=0)
    elif mode == 'max':
        val_fromsmp, idx = torch.max(all_dist, dim=1)
        val, idx_mmd = torch.max(all_dist, dim=0)
    mmd = val.mean()
    mmd_smp = val_fromsmp.mean()
    cov = float(idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
        'idx_mmd': idx_mmd,
        'mmd_contrib': val,
    }


def compute_all_metrics(sample_pcs, ref_pcs, batch_size, accelerated_cd=False,
                        f1_threshold=0.001, cd_option=False, one_part_of_cd=False, emd_option=False, f1_option=False):
    results = {}

    M_rs_cd, M_rs_emd, M_rs_f1_score, M_rs_cd_left, M_rs_cd_right = _pairwise_EMD_CD_F1_SCORE(
        sample_pcs, ref_pcs, batch_size, accelerated_cd=accelerated_cd,
        f1_threshold=f1_threshold, cd_option=cd_option,
        one_part_of_cd=one_part_of_cd, emd_option=emd_option, f1_option=f1_option)

    if cd_option:
        res_cd = lgan_mmd_cov(M_rs_cd)
        results.update({
            "%s-CD" % k: v for k, v in res_cd.items()
        })

    if emd_option:
        res_emd = lgan_mmd_cov(M_rs_emd)
        results.update({
            "%s-EMD" % k: v for k, v in res_emd.items()
        })

    if f1_option:
        res_f1_score = lgan_mmd_cov(M_rs_f1_score, 'max')
        results.update({
            "%s-F1" % k: v for k, v in res_f1_score.items()
        })
    if one_part_of_cd:
        res_cd_left = lgan_mmd_cov(M_rs_cd_left)
        results.update({
            "%s-CD-left" % k: v for k, v in res_cd_left.items()
        })

        res_cd_right = lgan_mmd_cov(M_rs_cd_right)
        results.update({
            "%s-CD-right" % k: v for k, v in res_cd_right.items()
        })


    M_rr_cd, M_rr_emd, M_rr_f1_score, M_rr_cd_left, M_rr_cd_right = _pairwise_EMD_CD_F1_SCORE(
        ref_pcs, ref_pcs, batch_size, accelerated_cd=accelerated_cd,
        f1_threshold=f1_threshold, cd_option=cd_option,
        one_part_of_cd=one_part_of_cd, emd_option=emd_option, f1_option=f1_option)
    M_ss_cd, M_ss_emd, M_ss_f1_score, M_ss_cd_left, M_ss_cd_right = _pairwise_EMD_CD_F1_SCORE(
        sample_pcs, sample_pcs, batch_size, accelerated_cd=accelerated_cd,
        f1_threshold=f1_threshold, cd_option=cd_option,
        one_part_of_cd=one_part_of_cd, emd_option=emd_option, f1_option=f1_option)

    # 1-NN results
    if cd_option:
        one_nn_cd_res = knn(M_ss_cd, M_rs_cd, M_rr_cd, 1, sqrt=False)
        results.update({
                "1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if 'acc' in k
        })

    if emd_option:
        one_nn_emd_res = knn(M_ss_emd, M_rs_emd, M_rr_emd, 1, sqrt=False)
        results.update({
            "1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if 'acc' in k
        })

    if f1_option:
        one_nn_f1_score_res = knn(M_ss_f1_score, M_rs_f1_score, M_rr_f1_score, 1, sqrt=False)
        results.update({
            "1-NN-F1-%s" % k: v for k, v in one_nn_f1_score_res.items() if 'acc' in k
        })

    if one_part_of_cd:
        one_nn_cd_left = knn(M_ss_cd_left, M_rs_cd_left, M_rr_cd_left, 1, sqrt=False)
        results.update({
            "1-NN-CD-left-%s" % k: v for k, v in one_nn_cd_left.items() if 'acc' in k
        })

        one_nn_cd_right = knn(M_ss_cd_right, M_rs_cd_right, M_rr_cd_right, 1, sqrt=False)
        results.update({
            "1-NN-CD-right-%s" % k: v for k, v in one_nn_cd_right.items() if 'acc' in k
        })

    return results


#######################################################
# JSD : from https://github.com/optas/latent_3d_points
#######################################################
def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28):
    """Computes the JSD between two sets of point-clouds, as introduced in the paper
    ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    """
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(pclouds, grid_resolution, in_sphere=False, verbose=False):
    """Given a collection of point-clouds, estimate the entropy of the random variables
    corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    """
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    """another way of computing JSD"""

    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))


if __name__ == "__main__":
    B, N = 5, 2048
    x = torch.rand(B, N, 3)
    y = torch.rand(B, N, 3)

    result = compute_all_metrics(x, y, batch_size=5, accelerated_cd=False)
    print(result)

    result_2 = compute_all_metrics(y, x, batch_size=5, accelerated_cd=False)
    print(result_2)
