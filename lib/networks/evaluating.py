import os
from time import time
from sys import stdout

import h5py as h5
import numpy as np
import torch

from lib.networks.utils import AverageMeter
from lib.networks.utils import JSD, f_score
from lib.metrics.evaluation_metrics import compute_all_metrics, EMD_CD_F1, distChamferCUDA, emd_approx

def evaluate(iterator, model, loss_func, **kwargs):
    train_mode = kwargs.get('train_mode')
    util_mode = kwargs.get('util_mode')
    is_saving = kwargs.get('saving')
    if is_saving:
        #saving generated point clouds, ground-truth point clouds and sampled labels.
        clouds_fname = '{}_{}_{}_{}_clouds_{}.h5'.format(kwargs['model_name'][:-4],
                                                         iterator.dataset.part,
                                                         kwargs['cloud_size'],
                                                         kwargs['sampled_cloud_size'],
                                                         util_mode)
        clouds_fname = os.path.join(kwargs['logging_path'], clouds_fname)
        print(clouds_fname)
        clouds_file = h5.File(clouds_fname, 'w')
        sampled_clouds = clouds_file.create_dataset(
            'sampled_clouds',
            shape=(kwargs['N_sets'] * len(iterator.dataset), 3, kwargs['sampled_cloud_size']),
            dtype=np.float32
        )
        gt_clouds = clouds_file.create_dataset(
            'gt_clouds',
            shape=(kwargs['N_sets'] * len(iterator.dataset), 3, kwargs['cloud_size']),
            dtype=np.float32
        )
        sampled_labels = clouds_file.create_dataset(
            'sampled_labels',
            shape=(kwargs['N_sets'] * len(iterator.dataset), kwargs['cloud_size']),
            dtype=np.int8
        )
        if train_mode == 'p_rnvp_mc_g_rnvp_vae_ic':
            print('save images')
            image_clouds = clouds_file.create_dataset(
                'image_clouds',
                shape=(kwargs['N_sets'] * len(iterator.dataset), 4, 224, 224),
                dtype=np.float32
            )

    batch_time = AverageMeter()
    data_time = AverageMeter()
    inf_time = AverageMeter()

    if util_mode == 'training':
        LB = AverageMeter()
        PNLL = AverageMeter()
        GNLL = AverageMeter()
        GENT = AverageMeter()

    elif util_mode == 'autoencoding':
        gen_clouds_buf = []
        ref_clouds_buf = []
    elif util_mode == 'generating':
        gen_clouds_buf = []
        ref_clouds_buf = []
    elif util_mode == 'reconstruction':
        CD = AverageMeter()
        EMD = AverageMeter()
        # F1 = AverageMeter()
        F1_lst = [AverageMeter() for _ in range(len(kwargs['f1_threshold_lst']))]
        # for _ in len(kwargs['f1_threshold_lst']):
        #     F1 = AverageMeter()
        #     F1_lst.append(F1)

    model.eval()
    torch.set_grad_enabled(False)

    end = time()

    for i, batch in enumerate(iterator):
        data_time.update(time() - end)

        g_clouds = batch['cloud'].cuda(non_blocking=True)
        p_clouds = batch['eval_cloud'].cuda(non_blocking=True)

        inf_end = time()
        n_components = kwargs.get('n_components')
        n = kwargs.get('sampled_cloud_size')

        # for test, generate samples
        with torch.no_grad():
            if train_mode == 'p_rnvp_mc_g_rnvp_vae':
                output_prior, samples, labels, log_weights = model(g_clouds, p_clouds, images=None, n_sampled_points=n, labeled_samples=True)
            elif train_mode == 'p_rnvp_mc_g_rnvp_vae_ic':
                images = batch['image'].cuda(non_blocking=True)
                output_prior, samples, labels, log_weights = model(g_clouds, p_clouds, images, n_sampled_points=n, labeled_samples=True)
        
        inf_time.update((time() - inf_end) / g_clouds.shape[0], g_clouds.shape[0])

        r_clouds = samples
        if kwargs['unit_scale_evaluation']:
            if kwargs['cloud_scale']:
                r_clouds *= kwargs['cloud_scale_scale']
                p_clouds *= kwargs['cloud_scale_scale']
        if kwargs['orig_scale_evaluation']:
            if kwargs['cloud_scale']:
                r_clouds *= kwargs['cloud_scale_scale']
                p_clouds *= kwargs['cloud_scale_scale']

            if kwargs['cloud_translate']:
                shift = torch.from_numpy(np.array(kwargs['cloud_translate_shift']).reshape(1, -1, 1)).cuda()
                r_clouds += shift
                p_clouds += shift

            if not kwargs['cloud_rescale2orig']:
                r_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                p_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
            if not kwargs['cloud_recenter2orig']:
                r_clouds += batch['orig_c'].unsqueeze(2).cuda()
                p_clouds += batch['orig_c'].unsqueeze(2).cuda()

        if is_saving:
            # saving generated point clouds, ground-truth point clouds and sampled labels.
            sampled_clouds[kwargs['batch_size'] * i:kwargs['batch_size'] * i + r_clouds.shape[0]] = \
                r_clouds.detach().cpu().numpy().astype(np.float32)
            gt_clouds[kwargs['batch_size'] * i:kwargs['batch_size'] * i + p_clouds.shape[0]] = \
                p_clouds.detach().cpu().numpy().astype(np.float32)
            sampled_labels[kwargs['batch_size'] * i:kwargs['batch_size'] * i + p_clouds.shape[0]] = \
                labels.detach().cpu().numpy().astype(np.int)
            if train_mode == 'p_rnvp_mc_g_rnvp_vae_ic':
                # if have images, save images
                image_clouds[kwargs['batch_size'] * i:kwargs['batch_size'] * i + images.shape[0]] = \
                    images.cpu().numpy().astype(np.float32)
        
        if util_mode == 'autoencoding' or util_mode == 'generating':
            gen_clouds_buf.append(r_clouds)
            ref_clouds_buf.append(p_clouds)
        elif util_mode == 'reconstruction':
            # when do svr reconstruction, we compute results over batches, becuase its data size is too large.
            r_clouds = torch.transpose(r_clouds, 1, 2).contiguous()
            p_clouds = torch.transpose(p_clouds, 1, 2).contiguous()

            # line = ''
            if kwargs['cd']:
                dl, dr = distChamferCUDA(r_clouds, p_clouds)
                cd = (dl.mean(1) + dr.mean(1)).mean()
                CD.update(cd.item(), p_clouds.shape[0])
                # line += '\tCD {CD.val:.6f} ({CD.avg:.6f})'.format(CD=CD)
            if kwargs['emd']:
                emd = emd_approx(r_clouds, p_clouds).mean()
                EMD.update(emd.item(), p_clouds.shape[0])
                # line += '\tEMD {EMD.val:.6f} ({EMD.avg:.6f})'.format(EMD=EMD)
            if kwargs['f1']:
                for i, f1_threshold in enumerate(kwargs['f1_threshold_lst']):
                    f1 = f_score(r_clouds, p_clouds, threshold=f1_threshold).mean()
                    F1_lst[i].update(f1.item(), p_clouds.shape[0])
                    # line += '\tF1 {F1.val:.1f} ({F1.avg:.1f})'.format(F1=F1_lst[i])
            # line += '\n'
            # stdout.write(line)
            # stdout.flush()

        batch_time.update(time() - end)
        end = time()

    print('Inference time: {} sec/sample'.format(inf_time.avg))

    if util_mode == 'autoencoding':
        # compute cd, emd and f1 for auto-encodings
        gen_clouds_buf = torch.transpose(torch.cat(gen_clouds_buf, dim=0), 2, 1).contiguous()
        ref_clouds_buf = torch.transpose(torch.cat(ref_clouds_buf, dim=0), 2, 1).contiguous()
        res = {}
        for _, f1_threshold in enumerate(kwargs['f1_threshold_lst']):
            metrics = EMD_CD_F1(gen_clouds_buf, ref_clouds_buf, batch_size=60, accelerated_cd=True, reduced=True,
                                cd_option=kwargs['cd'], emd_option=kwargs['emd'],
                                one_part_of_cd=False,
                                f1_option=kwargs['f1'], f1_threshold=f1_threshold)

            if kwargs['cd']:
                cd = metrics['CD'] * 1e4
                print('CD:\t{:.2f}'.format(cd))
                res['cd'] = cd
            if kwargs['emd']:
                emd = metrics['EMD'] * 1e2
                print('EMD:\t{:.2f}'.format(emd))
                res['emd'] = emd
            if kwargs['f1']:
                f1 = metrics['F1']
                print('F1-%.4f: %.2f' % (f1_threshold, f1))
                res['f1_%.4f' % (f1_threshold)] = f1

    elif util_mode == 'generating':
        # compute mmd-cd, mmd-emd, mmd-f1 for generation task
        gen_clouds_buf = torch.transpose(torch.cat(gen_clouds_buf, dim=0), 2, 1).contiguous()
        ref_clouds_buf = torch.transpose(torch.cat(ref_clouds_buf, dim=0), 2, 1).contiguous()

        gen_clouds_buf = gen_clouds_buf.cpu().numpy()
        gen_clouds_inds = set(np.arange(gen_clouds_buf.shape[0]))
        nan_gen_clouds_inds = set(np.isnan(gen_clouds_buf).sum(axis=(1, 2)).nonzero()[0])
        gen_clouds_inds = list(gen_clouds_inds - nan_gen_clouds_inds)
        dup_gen_clouds_inds = np.random.choice(gen_clouds_inds, size=len(nan_gen_clouds_inds))
        gen_clouds_buf[list(nan_gen_clouds_inds)] = gen_clouds_buf[dup_gen_clouds_inds]
        gen_clouds_buf = torch.from_numpy(gen_clouds_buf).cuda()

        res = {}
        if kwargs['jsd']:
            jsd = JSD(gen_clouds_buf.cpu().numpy(), ref_clouds_buf.cpu().numpy(),
                      clouds1_flag='gen', clouds2_flag='ref', warning=False)
            jsd = jsd * 1e2
            print('JSD:\t{:.2f}'.format(jsd))
            res['jsd'] = jsd

        for _, f1_threshold in enumerate(kwargs['f1_threshold_lst']):
            metrics = compute_all_metrics(
                gen_clouds_buf, ref_clouds_buf, batch_size=60, accelerated_cd=True,
                f1_threshold=f1_threshold, cd_option=kwargs['cd'], one_part_of_cd=False,
                emd_option=kwargs['emd'], f1_option=kwargs['f1'])

            if kwargs['cd']:
                cd_mmds = metrics['lgan_mmd-CD'] * 1e4
                cd_covs = metrics['lgan_cov-CD'] * 1e2
                cd_1nns = metrics['1-NN-CD-acc'] * 1e2

                print('MMD-CD:\t{:.2f}'.format(cd_mmds))
                print('COV-CD:\t{:.2f}'.format(cd_covs))
                print('1NN-CD:\t{:.2f}'.format(cd_1nns))

                res['cd_mmds'] = cd_mmds
                res['cd_covs'] = cd_covs
                res['cd_1nns'] = cd_1nns
            if kwargs['emd']:
                emd_mmds = metrics['lgan_mmd-EMD'] * 1e2
                emd_covs = metrics['lgan_cov-EMD'] * 1e2
                emd_1nns = metrics['1-NN-EMD-acc'] * 1e2
                print('MMD-EMD:\t{:.2f}'.format(emd_mmds))
                print('COV-EMD:\t{:.2f}'.format(emd_covs))
                print('1NN-EMD:\t{:.2f}'.format(emd_1nns))

                res['emd_mmds'] = emd_mmds
                res['emd_covs'] = emd_covs
                res['emd_1nns'] = emd_1nns
            if kwargs['f1']:
                f1_mmds = metrics['lgan_mmd-F1']
                f1_covs = metrics['lgan_cov-F1'] * 1e2
                f1_1nns = metrics['1-NN-F1-acc'] * 1e2
                print('MMD-F1-%.4f: %.2f' % (f1_threshold, f1_mmds))
                print('COV-F1-%.4f: %.2f' % (f1_threshold, f1_covs))
                print('1NN-F1-%.4f: %.2f' % (f1_threshold, f1_1nns))
                res['f1_%.4f_mmds' % (f1_threshold)] = f1_mmds
                res['f1_%.4f_covs' % (f1_threshold)] = f1_covs
                res['f1_%.4f_1nns' % (f1_threshold)] = f1_1nns

    elif util_mode == 'reconstruction':
        # compute cd, emd adn f1 for reconstruction tasks
        if kwargs['cd']:
            print('CD: {:.6f}'.format(CD.avg))
        if kwargs['emd']:
            print('EMD: {:.6f}'.format(EMD.avg))
        if kwargs['f1']:
            for i, f1_threshold in enumerate(kwargs['f1_threshold_lst']):
                print('F1-%.4f: %.2f' % (f1_threshold, F1_lst[i].avg))
        res = [CD.avg, EMD.avg]

    if is_saving:
        clouds_file.close()

    return res

'''
def interpolate(iterator, model, **kwargs):
    saving_mode = kwargs.get('saving_mode')
    N_saved_batches = 3

    model.eval()
    torch.set_grad_enabled(False)

    if saving_mode:
        if kwargs['interpolate_one_flow']:
            clouds_fname = 'interpolate_one_flow_{}_{}_{}.h5'.format(kwargs['model_name'][:-4], iterator.dataset.part, kwargs['cloud_size'])
        else:
            clouds_fname = 'interpolation_{}_{}_{}.h5'.format(kwargs['model_name'][:-4], iterator.dataset.part, kwargs['cloud_size'])
        clouds_fname = kwargs['path2save'] + clouds_fname

        clouds_file = h5.File(clouds_fname, 'w')
        clouds1 = clouds_file.create_dataset(
            'clouds1',
            shape=(N_saved_batches * kwargs['batch_size'], 3, kwargs['cloud_size']),
            dtype=np.float32
        )
        clouds2 = clouds_file.create_dataset(
            'clouds2',
            shape=(N_saved_batches * kwargs['batch_size'], 3, kwargs['cloud_size']),
            dtype=np.float32
        )
        interpolations = clouds_file.create_dataset(
            'interpolations',
            shape=(N_saved_batches * kwargs['batch_size'], 3, kwargs['cloud_size'], 9),
            dtype=np.float32
        )
        inter_labels = clouds_file.create_dataset(
            'labels',
            shape=(N_saved_batches * kwargs['batch_size'], kwargs['cloud_size'], 9),
            dtype=np.uint8
        )
    for i, batch in enumerate(iterator):
        if i == N_saved_batches:
            break

        clouds = batch['cloud'].cuda(non_blocking=True)
        ref_clouds = batch['eval_cloud'].cuda(non_blocking=True)
        inds = np.arange(ref_clouds.shape[0])
        np.random.shuffle(inds)
        ref_clouds = ref_clouds[inds].contiguous()

        codes1 = model.encode(clouds)['g_posterior_mus']
        codes2 = model.encode(ref_clouds)['g_posterior_mus']
        n_components = kwargs['n_components']
        
        # logits_exp = np.exp(model.mixture_weights_logits.detach().cpu().numpy())
        # probs = logits_exp / logits_exp.sum()
        # flows_idx = np.random.choice(range(n_components), size=kwargs['cloud_size'], p=probs)
        # 
        # masks = []
        # for t in range(n_components):
        #     mask = flows_idx == t
        #     masks.append(mask.sum())
        # with torch.no_grad():
        #     output_decoder = model.decode(clouds, codes1, masks)
        # 
        # # samples = []
        # samples = torch.zeros_like((clouds))
        # labels = torch.zeros(clouds.size(0), clouds.size(2))
        # for t in range(n_components):
        #     s = output_decoder[t]
        #     mask = flows_idx == t
        #     #print(s['p_prior_samples'][-1].shape)
        #     samples[:, :, mask] = s['p_prior_samples'][-1]
        #     labels[:, mask] = t + 1
        
        ints = torch.from_numpy(np.zeros((clouds.shape) + (9,), dtype=np.float32))
        w = torch.from_numpy(np.float32(np.arange(1, 10, 1).reshape(1, 1, -1) / 10)).cuda()
        interpolated_codes = (1. - w) * codes1.unsqueeze(2) + w * codes2.unsqueeze(2)

        labels_itp = torch.zeros(clouds.size(0), clouds.size(2), 9, dtype=torch.uint8)
        pc_decoder = model.pc_decoder
        for j in range(9):
            if kwargs.get('interpolation') and kwargs.get('interpolate_one_flow'):
                m = 0
                output_decoder_itp = []

                mixture_weights_logits = model.get_weights(interpolated_codes[:, :, j])
                logits_exp = np.exp(mixture_weights_logits[0].detach().cpu().numpy())
                probs = logits_exp / logits_exp.sum()
                flows_idx = np.random.choice(range(n_components), size=kwargs['sampled_cloud_size'], p=probs)
                masks = []
                for t in range(n_components):
                    mask = flows_idx == t
                    masks.append(mask.sum())

                for idx in range(n_components):
                    if idx == kwargs['flow_idx'][m]:
                        one_decoder_itp = model.one_flow_decode(clouds, interpolated_codes[:, :, j], pc_decoder[idx], masks[idx])
                        m += 1
                    else:
                        one_decoder_itp = model.one_flow_decode(clouds, codes1, pc_decoder[idx], masks[idx])
                    output_decoder_itp.append(one_decoder_itp)
            else:
                output_decoder_itp = model.decode(clouds, interpolated_codes[:, :, j], masks)
            # samples = []
            samples_itp = torch.zeros_like(clouds)
            for t in range(n_components):
                s_itp = output_decoder_itp[t]
                mask = flows_idx == t
                samples_itp[:, :, mask] = s_itp['p_prior_samples'][-1]
                labels_itp[:, mask, j] = t + 1
            ints[:, :, :, j] = samples_itp
        if saving_mode:
            clouds1[kwargs['batch_size'] * i:kwargs['batch_size'] * i + clouds.shape[0]] = clouds.cpu().numpy()
            clouds2[kwargs['batch_size'] * i:kwargs['batch_size'] * i + clouds.shape[0]] = ref_clouds.cpu().numpy()
            interpolations[kwargs['batch_size'] * i:kwargs['batch_size'] * i + clouds.shape[0]] = ints.cpu().numpy()
            inter_labels[kwargs['batch_size'] * i:kwargs['batch_size'] * i + clouds.shape[0]] = labels_itp.cpu().numpy().astype(np.uint8)
    if saving_mode:
        clouds_file.close() 

def sample(iterator, model, **kwargs):
    saving_mode = kwargs.get('saving_mode')
    N_saved_batches = 10
    sparse_size = kwargs['sampled_cloud_size']

    model.eval()
    torch.set_grad_enabled(False)

    if saving_mode:
        clouds_fname = 'sample_{}_{}_{}.h5'.format(kwargs['model_name'][:-4], iterator.dataset.part, kwargs['cloud_size'])
        clouds_fname = kwargs['path2save'] + clouds_fname

        clouds_file = h5.File(clouds_fname, 'w')
        clouds_sparse = clouds_file.create_dataset(
            'clouds_sparse',
            shape=(N_saved_batches * kwargs['batch_size'], 3, sparse_size),
            dtype=np.float32
        )
        clouds_dense = clouds_file.create_dataset(
            'clouds_dense',
            shape=(N_saved_batches * kwargs['batch_size'], 3, kwargs['cloud_size']),
            dtype=np.float32
        )
        labels_dense = clouds_file.create_dataset(
            'labels',
            shape=(N_saved_batches * kwargs['batch_size'], kwargs['cloud_size']),
            dtype=np.uint8
        )

    for i, batch in enumerate(iterator):
        if i == N_saved_batches:
            break

        clouds = batch['cloud'].cuda(non_blocking=True)
        clouds = clouds[:, :, :sparse_size]

        codes = model.encode(clouds)['g_posterior_mus']
        n_components = kwargs['n_components']

        # logits_exp = np.exp(model.mixture_weights_logits.detach().cpu().numpy())
        # probs = logits_exp / logits_exp.sum()
        # flows_idx = np.random.choice(range(n_components), size=kwargs['cloud_size'], p=probs)
        # masks = []
        # for t in range(n_components):
        #     mask = flows_idx == t
        #     masks.append(mask.sum())
        # with torch.no_grad():
        #     output_decoder = model.decode(clouds, codes)
        # samples = torch.zeros(clouds.size(0), 3, kwargs['cloud_size']).cuda(non_blocking=True)
        # labels = torch.zeros(clouds.size(0), kwargs['cloud_size'])
        # for t in range(n_components):
        #     s = output_decoder[t]
        #     mask = flows_idx == t
        #     samples[:, :, mask] = s['p_prior_samples'][-1]
        #     labels[:, mask] = t + 1

        mixture_weights_logits = model.get_weights(codes)
        logits_exp = np.exp(mixture_weights_logits[0].detach().cpu().numpy())
        probs = logits_exp / logits_exp.sum()
        flows_idx = np.random.choice(range(n_components), size=kwargs['sampled_cloud_size'], p=probs)
        masks = []
        for t in range(n_components):
            mask = flows_idx == t
            masks.append(mask.sum())
        with torch.no_grad():
            samples, labels, _ = model.decode(clouds, codes, n_sampled_points=masks, labeled_samples=True)


        if saving_mode:
            clouds_sparse[kwargs['batch_size'] * i:kwargs['batch_size'] * i + clouds.shape[0]] = clouds.cpu().numpy()
            clouds_dense[kwargs['batch_size'] * i:kwargs['batch_size'] * i + samples.shape[0]] = samples.cpu().numpy()
            labels_dense[kwargs['batch_size'] * i:kwargs['batch_size'] * i + labels.shape[0]] = labels.cpu().numpy()
    if saving_mode:    
        clouds_file.close()
'''
