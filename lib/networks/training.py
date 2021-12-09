import os
import time
from sys import stdout
import torch
import h5py as h5
import numpy as np

from lib.networks.utils import AverageMeter, save_model
from lib.visualization.utils import add_figures_reconstruction_tb, add_svr_reconstruction_tb


def train(iterator, model, loss_func, optimizer, scheduler, epoch, iter, warmup, train_writer, **kwargs):
    num_workers = kwargs.get('num_workers')
    train_mode = kwargs.get('train_mode')
    model_name = os.path.join(kwargs['logging_path'], kwargs.get('model_name'))

    batch_time = AverageMeter()
    data_time = AverageMeter()

    LB = AverageMeter()
    PNLL = AverageMeter()
    GNLL = AverageMeter()
    GENT = AverageMeter()

    model.train()
    torch.set_grad_enabled(True)

    end = time.time()

    for i, batch in enumerate(iterator):
        if iter + i >= len(iterator):
            break
        data_time.update(time.time() - end)
        scheduler(optimizer, epoch, iter + i)

        g_clouds = batch['cloud'].cuda(non_blocking=True)
        p_clouds = batch['eval_cloud'].cuda(non_blocking=True)
        # returns shape distributions list in prior flows, samples list in decoder flows
        # and log weights of all flows in decoder flows.
        output_prior, output_decoder, mixture_weights_logits = model(g_clouds, p_clouds, images=None, n_sampled_points=None, labeled_samples=False, warmup=warmup)

        loss, pnll, gnll, gent = loss_func(output_prior, output_decoder, mixture_weights_logits)
        with torch.no_grad():
            if torch.isnan(loss):
                print('Loss is NaN! Stopping without updating the net...')
                exit()

        PNLL.update(pnll.item(), g_clouds.shape[0])
        GNLL.update(gnll.item(), g_clouds.shape[0])
        GENT.update(gent.item(), g_clouds.shape[0])
        LB.update((pnll + gnll - gent).item(), g_clouds.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        if (iter + i + 1) % (num_workers) == 0 and kwargs['logging']:
            line = 'Epoch: [{0}][{1}/{2}]'.format(epoch + 1, iter + i + 1, len(iterator))
            line += '\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(batch_time=batch_time)
            line += '\tLB {LB.val:.2f} ({LB.avg:.2f})'.format(LB=LB)
            line += '\tPNLL {PNLL.val:.2f} ({PNLL.avg:.2f})'.format(PNLL=PNLL)
            line += '\tGNLL {GNLL.val:.2f} ({GNLL.avg:.2f})'.format(GNLL=GNLL)
            line += '\tGENT {GENT.val:.2f} ({GENT.avg:.2f})'.format(GENT=GENT)
            line += '\n'
            stdout.write(line)
            stdout.flush()

        end = time.time()

        if (iter + i + 1) % (100 * num_workers) == 0 and kwargs['logging']:
            if kwargs['distributed']:
                sd = model.module.state_dict()
            else:
                sd = model.state_dict()
            save_model({
                'epoch': epoch,
                'iter': iter + i + 1,
                'model_state': sd,
                'optimizer_state': optimizer.state_dict()
            }, model_name)

    # write to tensorboard
    if kwargs['logging']:
        train_writer.add_scalar('train/loss', LB.avg, epoch)
        train_writer.add_scalar('train/PNLL', PNLL.avg, epoch)
        train_writer.add_scalar('train/GNLL', GNLL.avg, epoch)
        train_writer.add_scalar('train/GENT', GENT.avg, epoch)

    if kwargs['logging']:
        if kwargs['distributed']:
            sd = model.module.state_dict()
        else:
            sd = model.state_dict()
        save_model({
            'epoch': epoch + 1,
            'iter': 0,
            'model_state': sd,
            'optimizer_state': optimizer.state_dict()
        }, model_name)


def eval(iterator, model, loss_func, optimizer, epoch, iter, warmup, min_loss, eval_writer, **kwargs):
    train_mode = kwargs.get('train_mode')

    LB = AverageMeter()
    PNLL = AverageMeter()
    GNLL = AverageMeter()
    GENT = AverageMeter()

    model.eval()
    torch.set_grad_enabled(False)

    for i, batch in enumerate(iterator):
        if iter + i >= len(iterator):
            break
        g_clouds = batch['cloud'].cuda(non_blocking=True)
        p_clouds = batch['eval_cloud'].cuda(non_blocking=True)
        output_prior, output_decoder, mixture_weights_logits = model(g_clouds, p_clouds, images=None, n_sampled_points=None, labeled_samples=False, warmup=warmup)

        with torch.no_grad():
            loss, pnll, gnll, gent = loss_func(output_prior, output_decoder, mixture_weights_logits)

        PNLL.update(pnll.item(), g_clouds.shape[0])
        GNLL.update(gnll.item(), g_clouds.shape[0])
        GENT.update(gent.item(), g_clouds.shape[0])
        LB.update((pnll + gnll - gent).item(), g_clouds.shape[0])

        with torch.no_grad():
            if torch.isnan(loss):
                print('Loss is NaN! Stopping without updating the net...')
                exit()
            if torch.isinf(loss):
                print('Loss is INF! Stopping without updating the net...')
                exit()

    if kwargs.get('logging'):
        print('[epoch %d]: eval loss %f' % (epoch, LB.avg))

    # write to tensorboard
    if kwargs.get('logging'):
        eval_writer.add_scalar('val/loss', LB.avg, epoch)
        eval_writer.add_scalar('val/PNLL', PNLL.avg, epoch)
        eval_writer.add_scalar('val/GNLL', GNLL.avg, epoch)
        eval_writer.add_scalar('val/GENT', GENT.avg, epoch)

    # Add reconstruction visualization to tensorboard
    if kwargs.get('logging_img') and epoch % kwargs.get('logging_img_frequency') == 0 and kwargs.get('logging'):
        npy_path = os.path.join(kwargs.get('logging_path'), '')

        if kwargs.get('distributed'):
            tmp_mode = model.module.mode
            model.module.mode = 'autoencoding'
            all_samples, all_gts, all_labels = reconstruct(iterator, model, max_batches=1, warmup=False, **kwargs)
            model.module.mode = tmp_mode
        else:
            tmp_mode = model.mode
            model.mode = 'autoencoding'
            all_samples, all_gts, all_labels = reconstruct(iterator, model, max_batches=1, warmup=False, **kwargs)
            model.mode = tmp_mode

        # save numpy data
        all_labels = all_labels.detach().cpu().numpy()
        all_samples = all_samples.detach().cpu().numpy()
        all_gts = all_gts.detach().cpu().numpy()

        add_figures_reconstruction_tb(all_gts, all_samples, all_labels, eval_writer, epoch)

    if LB.avg < min_loss:
        min_loss = LB.avg
        best_modelname = 'best_model_' + kwargs.get('model_name')
        best_model_name = os.path.join(kwargs['logging_path'], best_modelname)
        if kwargs.get('logging'):
            if kwargs['distributed']:
                sd = model.module.state_dict()
            else:
                sd = model.state_dict()
            save_model({
                'epoch': epoch + 1,
                'iter': 0,
                'model_state': sd,
                'optimizer_state': optimizer.state_dict()
            }, best_model_name)
    return min_loss



def train_svr(iterator, model, loss_func, optimizer, scheduler, epoch, iter, warmup, train_writer, **kwargs):
    num_workers = kwargs.get('num_workers')
    train_mode = kwargs.get('train_mode')
    model_name = os.path.join(kwargs['logging_path'], kwargs.get('model_name'))

    batch_time = AverageMeter()
    data_time = AverageMeter()

    LB = AverageMeter()
    PNLL = AverageMeter()
    GNLL = AverageMeter()
    GENT = AverageMeter()

    model.train()
    torch.set_grad_enabled(True)

    end = time.time()

    for i, batch in enumerate(iterator):
        if iter + i >= len(iterator):
            break
        data_time.update(time.time() - end)
        scheduler(optimizer, epoch, iter + i)

        g_clouds = batch['cloud'].cuda(non_blocking=True)
        p_clouds = batch['eval_cloud'].cuda(non_blocking=True)
        images = batch['image'].cuda(non_blocking=True)
        output_prior, output_decoder, mixture_weights_logits = model(g_clouds, p_clouds, images,
                            n_sampled_points=None, labeled_samples=False, warmup=warmup)

        loss, pnll, gnll, gent = loss_func(output_prior, output_decoder, mixture_weights_logits)

        with torch.no_grad():
            if torch.isnan(loss):
                print('Loss is NaN! Stopping without updating the net...')
                exit()

        PNLL.update(pnll.item(), g_clouds.shape[0])
        GNLL.update(gnll.item(), g_clouds.shape[0])
        GENT.update(gent.item(), g_clouds.shape[0])
        LB.update((pnll + gnll - gent).item(), g_clouds.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        if (iter + i + 1) % (num_workers) == 0 and kwargs['logging']:
            line = 'Epoch: [{0}][{1}/{2}]'.format(epoch + 1, iter + i + 1, len(iterator))
            line += '\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(batch_time=batch_time)
            line += '\tLB {LB.val:.2f} ({LB.avg:.2f})'.format(LB=LB)
            line += '\tPNLL {PNLL.val:.2f} ({PNLL.avg:.2f})'.format(PNLL=PNLL)
            line += '\tGNLL {GNLL.val:.2f} ({GNLL.avg:.2f})'.format(GNLL=GNLL)
            line += '\tGENT {GENT.val:.2f} ({GENT.avg:.2f})'.format(GENT=GENT)
            line += '\n'
            stdout.write(line)
            stdout.flush()

        end = time.time()

        # write to tensorboard
        if kwargs.get('logging'):
            step = epoch * len(iterator) + iter + i + 1
            train_writer.add_scalar('train/loss', LB.avg, step)
            train_writer.add_scalar('train/PNLL', PNLL.avg, step)
            train_writer.add_scalar('train/GNLL', GNLL.avg, step)
            train_writer.add_scalar('train/GENT', GENT.avg, step)

        if (iter + i + 1) % (100 * num_workers) == 0 and kwargs.get('logging'):
            # Add reconstruction visualization to tensorboard
            if kwargs['distributed']:
                sd = model.module.state_dict()
            else:
                sd = model.state_dict()
            save_model({
                'epoch': epoch,
                'iter': iter + i + 1,
                'model_state': sd,
                'optimizer_state': optimizer.state_dict()
            }, model_name)

            if kwargs.get('logging_img'):
                npy_path = os.path.join(kwargs.get('logging_path'), '')

                if kwargs.get('distributed'):
                    tmp_mode = model.module.mode
                    model.module.mode = 'reconstruction'
                    all_samples, all_gts, all_labels, all_images = reconstruct(iterator, model,
                                                            max_batches=1, warmup=warmup, **kwargs)
                    model.module.mode = tmp_mode
                else:
                    tmp_mode = model.mode
                    model.mode = 'reconstruction'
                    all_samples, all_gts, all_labels, all_images = reconstruct(iterator, model,
                                                            max_batches=1, warmup=warmup, **kwargs)
                    model.mode = tmp_mode

                # save numpy data
                all_labels = all_labels.detach().cpu().numpy()
                all_samples = all_samples.detach().cpu().numpy()
                all_gts = all_gts.detach().cpu().numpy()
                all_images = all_images.detach().cpu().numpy()
                all_images = all_images.transpose(0, 2, 3, 1)
                add_svr_reconstruction_tb(all_images, all_gts, all_samples, all_labels, train_writer, step)
                
    if kwargs['logging']:
        if kwargs['distributed']:
            sd = model.module.state_dict()
        else:
            sd = model.state_dict()
        save_model({
            'epoch': epoch + 1,
            'iter': 0,
            'model_state': sd,
            'optimizer_state': optimizer.state_dict()
        }, model_name)


def save_point_clouds(batch_i, gt_cloud, gen_cloud, len_dataset, **kwargs):

    clouds_fname = '{}_{}_{}_segs_clouds.h5'.format(kwargs['model_name'][:-4],
                                                    kwargs['cloud_size'],
                                                    kwargs['util_mode'])
    cloud_fname = os.path.join(kwargs['experiment_path'], clouds_fname)
    if not os.path.exists(cloud_fname):
        clouds_file = h5.File(cloud_fname, 'w')
        sampled_clouds = clouds_file.create_dataset(
            'sampled_clouds',
            shape=(len_dataset, 3, kwargs['cloud_size']),dtype=np.float32)
        gt_clouds = clouds_file.create_dataset(
            'gt_clouds',
            shape=(len_dataset, 3, kwargs['cloud_size']),dtype=np.float32)
    else:
        clouds_file = h5.File(cloud_fname, 'a')
        sampled_clouds = clouds_file['sampled_clouds']
        gt_clouds = clouds_file['gt_clouds']

    sampled_clouds[kwargs['batch_size'] * batch_i:kwargs['batch_size'] * batch_i + gen_cloud.shape[0]] =\
        gen_cloud.cpu().numpy().astype(np.float32)

    gt_clouds[kwargs['batch_size'] * batch_i:kwargs['batch_size'] * batch_i + gt_cloud.shape[0]] =\
        gt_cloud.cpu().numpy().astype(np.float32)

    clouds_file.close()


def reconstruct(test_dataloader, model, max_batches=np.infty, warmup=False, **config):
    train_mode = config.get('train_mode')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_samples = []
    all_labels = []
    all_gts = []
    all_images = []
    for i, data in enumerate(test_dataloader):
        if i >= max_batches:
            break

        g_input = data['cloud'].cuda(non_blocking=True)
        p_input = data['eval_cloud'].cuda(non_blocking=True)

        n_components = config.get('n_components')
        n = config.get('cloud_size')

        with torch.no_grad():
            for j in range(g_input.shape[0]):
                if train_mode == 'p_rnvp_mc_g_rnvp_vae':
                    output_prior, samples, labels, log_weights = model(g_input[j].unsqueeze(0), p_input[j].unsqueeze(0),
                        images=None, n_sampled_points=n, labeled_samples=True, warmup=warmup)
                elif train_mode == 'p_rnvp_mc_g_rnvp_vae_ic':
                    images = data['image'].cuda(non_blocking=True)
                    output_prior, samples, labels, log_weights = model(g_input[j].unsqueeze(0), p_input[j].unsqueeze(0),
                        images[j].unsqueeze(0), n_sampled_points=n, labeled_samples=True, warmup=warmup)
                all_samples.append(samples)
                all_gts.append(p_input)
                all_labels.append(labels)
                if train_mode == 'p_rnvp_mc_g_rnvp_vae_ic':
                    all_images.append(images)

    all_samples = torch.cat(all_samples, dim=0)
    all_labels = torch.cat(all_labels, dim=0).reshape(-1, config.get('cloud_size'))
    all_gts = torch.cat(all_gts, dim=0)
    if train_mode == 'p_rnvp_mc_g_rnvp_vae_ic':
        all_images = torch.cat(all_images, dim=0)

    if train_mode == 'p_rnvp_mc_g_rnvp_vae':
        return all_samples, all_gts, all_labels
    elif train_mode == 'p_rnvp_mc_g_rnvp_vae_ic':
        return all_samples, all_gts, all_labels, all_images


def predict(test_dataloader, model, **config):
    torch.set_grad_enabled(False)
    model.eval()

    all_samples, all_gts, all_labels = reconstruct(test_dataloader, model, **config)

    print(all_samples.shape)
    print(all_labels.shape)
    print(all_gts.shape)
    np.save(os.path.join(config['experiment_path'], 'all_labels.npy'), all_labels.detach().cpu().numpy())
    np.save(os.path.join(config['experiment_path'], 'all_gts.npy'), all_gts.detach().cpu().numpy())
    np.save(os.path.join(config['experiment_path'], 'all_samples.npy'), all_samples.detach().cpu().numpy())
