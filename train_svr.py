import os
import io
import yaml
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from lib.datasets.datasets import ShapeNetAllDataset
from lib.datasets.image_transformations import ComposeImageTransformation
from lib.datasets.cloud_transformations import ComposeCloudTransformation
from lib.networks.losses import Flow_Mixture_Loss
from lib.networks.flow_mixture import Flow_Mixture_SVR_Model
from lib.networks.optimizers import Adam, LRUpdater
from lib.networks.training import train_svr
from lib.networks.utils import cnt_params
from datetime import datetime

def define_options_parser():
    parser = argparse.ArgumentParser(description='Model training script. Provide a suitable config.')
    parser.add_argument('config', type=str, help='Path to config file in YAML format.')
    parser.add_argument('modelname', type=str, help='Model name to save checkpoints.')
    parser.add_argument('n_epochs', type=int, help='Total number of training epochs.')
    parser.add_argument('lr', type=float, help='Learining rate value.')
    parser.add_argument('--weights_type', type=str, default='global_weights',
                        help='choose to use global_weights/learned_weights.')
    parser.add_argument('--warmup_epoch', type=int, default=5, help='epochs use global_weights.')
    parser.add_argument('--jobid', type=str, default='1',
                        help='Id of training. If empty we give new id based of datetime.')

    parser.add_argument('--resume', action='store_true',
                        help='Flag signaling if training is resumed from a checkpoint.')
    parser.add_argument('--resume_optimizer', action='store_true',
                        help='Flag signaling if optimizer parameters are resumed from a checkpoint.')

    parser.add_argument('--distributed', action='store_true',
                        help='Flag if use distributed training')
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    return parser

def main_worker(gpu, ngpus_per_node, args):
    #parser = define_options_parser()
    #args = parser.parse_args()
    with io.open(args.config, 'r') as stream:
        config = yaml.load(stream)
    config['jobid'] = args.jobid
    if not 'logging_path' in config.keys():
        name_extension = config['jobid'] if config['jobid'] != '' else datetime.now().strftime("%Y%m%d_%H%M%S")
        config['logging_path'] = os.path.join(config['path2save'], args.modelname + '_' + name_extension)
        with open(args.config, 'w') as outfile:
            yaml.dump(config, outfile)
    if not os.path.exists(config['logging_path']):
        os.makedirs(config['logging_path'])
    config['model_name'] = '{0}.pkl'.format(args.modelname)
    config['n_epochs'] = args.n_epochs
    config['min_lr'] = config['max_lr'] = args.lr
    config['resume'] = True if args.resume else False
    config['resume_optimizer'] = True if args.resume_optimizer else False
    config['distributed'] = True if args.distributed else False
    config['logging'] = not args.distributed or (args.distributed and gpu == 0)
    config['weights_type'] = args.weights_type
    print('Configurations loaded.')

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.world_size = args.gpus * args.nodes
        args.rank = args.nr * args.gpus + gpu
        torch.distributed.init_process_group(
            'nccl',init_method='env://', world_size=args.world_size, rank=args.rank)
        print("world_size: ", args.world_size)
        print("rank: ", args.rank)

        config['batch_size'] = config['batch_size'] // args.world_size + int(
                            config['batch_size'] % args.world_size > gpu)
        print('Distributed training runs on GPU {} with batch size {}'.format(gpu, config['batch_size']))

    if not os.path.exists(os.path.join(config['logging_path'], 'config.yaml')):
        with open(os.path.join(config['logging_path'], 'config.yaml'), 'w') as outfile:
            yaml.dump(config, outfile)

    image_transform = ComposeImageTransformation(**config)
    cloud_transform, _ = ComposeCloudTransformation(**config)
    train_dataset = ShapeNetAllDataset(config['path2data'], part='train',
                                       images_fname=config['images_fname'], meshes_fname=config['meshes_fname'],
                                       cloud_size=config['cloud_size'], return_eval_cloud=True,
                                       return_original_scale=config['cloud_rescale2orig'],
                                       image_transform=image_transform, cloud_transform=cloud_transform,
                                       chosen_label=config['chosen_label'])
    print('Dataset init: done.')
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank)
        train_iterator = DataLoader(
            dataset=train_dataset, batch_size=config['batch_size'], shuffle=False,
            num_workers=config['num_workers'], pin_memory=True, drop_last=True, sampler=train_sampler)
    else:
        train_iterator = DataLoader(
            dataset=train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'],
            num_workers=config['num_workers'], pin_memory=True, drop_last=True)

    print(f'Size of training data: {len(train_dataset)}')

    torch.cuda.set_device(gpu)
    model = Flow_Mixture_SVR_Model(**config)
    model = model.cuda(gpu)
    print('Model init done on GPU {}.'.format(gpu))
    print('Total number of parameters: {} on GPU {}'.format(cnt_params(model.parameters()), gpu))
    print('Total number of parameters in decoder flows: {}'.format(cnt_params(model.pc_decoder.parameters())))
    print('Model init: done.')

    criterion = Flow_Mixture_Loss(**config).cuda(gpu)
    print('Loss init: done on GPU {}.'.format(gpu))

    optimizer = Adam(model.parameters(), lr=config['max_lr'], weight_decay=config['wd'],
                     betas=(config['beta1'], config['max_beta2']), amsgrad=True)
    scheduler = LRUpdater(len(train_iterator), **config)
    print('Optimizer init: done on GPU {}'.format(gpu))

    if not config['resume']:
        cur_epoch = 0
        cur_iter = 0
    else:
        path2checkpoint = os.path.join(config['logging_path'], config['model_name'])
        checkpoint = torch.load(path2checkpoint, map_location='cpu')
        cur_epoch = checkpoint['epoch']
        cur_iter = checkpoint['iter']
        model.load_state_dict(checkpoint['model_state'])
        if config['resume_optimizer']:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        del checkpoint
        print('Model {} loaded.'.format(path2checkpoint))

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    print("training")
    # configure tensorboard logging
    summary_writer = None
    if gpu == 0:
        tb_path = os.path.join(config['logging_path'], 'log')
        summary_writer = SummaryWriter(tb_path)

    for epoch in range(cur_epoch, config['n_epochs']):
        warmup = True if epoch < args.warmup_epoch else False
        train_svr(train_iterator, model, criterion, optimizer, scheduler, epoch, cur_iter, warmup, summary_writer, **config)
        cur_iter = 0

    summary_writer.close()

def main():
    parser = define_options_parser()
    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()
    if args.distributed and ngpus_per_node > 1:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '5555'
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(0, 1, args)


if __name__ == '__main__':
    main()
