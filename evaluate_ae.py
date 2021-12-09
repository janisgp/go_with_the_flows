import argparse
import os
import io
import yaml

import torch
from torch.utils.data import DataLoader
from lib.datasets.datasets import ShapeNetAllDataset
from lib.datasets.datasets import ShapeNetCoreDataset
from lib.datasets.cloud_transformations import ComposeCloudTransformation
from lib.datasets.image_transformations import ComposeImageTransformation
from lib.networks.evaluating import evaluate #, interpolate, sample
from lib.networks.flow_mixture import Flow_Mixture_Model, Flow_Mixture_SVR_Model
from lib.networks.losses import Flow_Mixture_Loss


def define_options_parser():
    parser = argparse.ArgumentParser(description='Model training script. Provide a suitable config.')
    parser.add_argument('experiment_path', type=str, help='Path to experiment.')
    parser.add_argument('modelname', type=str, help='Model name (without ending).')
    parser.add_argument('part', help='Part of dataset (train / val / test).')
    parser.add_argument('cloud_size', type=int, help='Number of input points.')
    parser.add_argument('sampled_cloud_size', type=int, help='Number of sampled points.')
    parser.add_argument('mode', type=str, help='Prediction mode (training / evaluating / generating / predicting).')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='choose to compute metrics of f1')
    parser.add_argument('--weights_type', type=str, default='global_weights',
                        help='choose to use global_weights/learned_weights.')
    parser.add_argument('--reps', type=int, default=10,
                        help='Repetitions of generating evaluations.')
    parser.add_argument('--orig_scale_evaluation', action='store_true',
                        help='Evaluation in original per cloud scale flag.')
    parser.add_argument('--unit_scale_evaluation', action='store_true',
                        help='Evaluation in unit scaoe per cloud scale flag')
    parser.add_argument('--save', action='store_true',
                        help='Saving flag.')
    parser.add_argument('--f1_threshold_lst', type=lambda s: [float(item) for item in s.split(',')], default=[0.0001],
                        help='the threshold to compute f1')
    parser.add_argument('--jsd', action='store_true',
                         help='choose t compute jsd or not')
    parser.add_argument('--cd', action='store_true',
                        help='choose to compute cd or not')
    parser.add_argument('--emd', action='store_true',
                        help='choose to compute metrics of emd')
    parser.add_argument('--f1', action='store_true',
                        help='choose to compute metrics of f1')
    '''
    parser.add_argument('--interpolation', action='store_true',
                        help='choose to do interpolation')
    parser.add_argument('--sampling', action='store_true',
                        help='choose to do sampling')
    parser.add_argument('--interpolate_one_flow', action='store_true',
                        help='choose to do interpolation on one flow')
    parser.add_argument('--flow_idx', default=0, type=int,
                        help='flow idx to do interpolation on')
    '''
    return parser


parser = define_options_parser()
args = parser.parse_args()
with io.open(os.path.join(args.experiment_path, 'config.yaml'), 'r') as stream:
    config = yaml.load(stream)
config['model_name'] = '{0}.pkl'.format(args.modelname)
config['experiment_path'] = args.experiment_path
config['part'] = args.part
config['cloud_size'] = args.cloud_size
config['sampled_cloud_size'] = args.sampled_cloud_size
config['util_mode'] = args.mode
config['orig_scale_evaluation'] = True if args.orig_scale_evaluation else False
config['unit_scale_evaluation'] = True if args.unit_scale_evaluation else False
config['saving_mode'] = True if args.save else False
config['N_sets'] = 1

config['f1_threshold_lst'] = args.f1_threshold_lst
config['jsd'] = True if args.jsd else False
config['cd'] = True if args.cd else False
#config['one_part_of_cd'] = True if args.one_part_of_cd else False
config['emd'] = True if args.emd else False
config['f1'] = True if args.f1 else False
#config['interpolation'] = True if args.interpolation else False
#config['interpolate_one_flow'] = True if args.interpolate_one_flow else False
#config['flow_idx'] = [0, 3]
config['weights_type'] = args.weights_type
print('Configurations loaded.')

if config['train_mode'] == 'p_rnvp_mc_g_rnvp_vae':
    cloud_transform, _ = ComposeCloudTransformation(**config)
    eval_dataset = ShapeNetCoreDataset(config['path2data'],
                                   part=args.part, meshes_fname=config['meshes_fname'],
                                   cloud_size=config['cloud_size'], return_eval_cloud=True,
                                   return_original_scale=config['cloud_rescale2orig'] or config['orig_scale_evaluation'],
                                   cloud_transform=cloud_transform,
                                   chosen_label=config['chosen_label'])
elif config['train_mode'] == 'p_rnvp_mc_g_rnvp_vae_ic':
    image_transform = ComposeImageTransformation(**config)
    cloud_transform, _ = ComposeCloudTransformation(**config)
    eval_dataset = ShapeNetAllDataset(config['path2data'], part=args.part,
                                       images_fname=config['images_fname'], meshes_fname=config['meshes_fname'],
                                       cloud_size=config['cloud_size'], return_eval_cloud=True,
                                       return_original_scale=config['cloud_rescale2orig'] or config['orig_scale_evaluation'],
                                       image_transform=image_transform, cloud_transform=cloud_transform,
                                       chosen_label=config['chosen_label'])
print('Dataset init: done.')

eval_iterator = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=config['num_workers'], pin_memory=True, drop_last=False)
print(len(eval_iterator))
print('Iterator init: done.')

if config['train_mode'] == 'p_rnvp_mc_g_rnvp_vae':
    model = Flow_Mixture_Model(**config).cuda()
elif config['train_mode'] == 'p_rnvp_mc_g_rnvp_vae_ic':
    model = Flow_Mixture_SVR_Model(**config).cuda()
else:
  raise ValueError(f'Unknown train_mode {config["train_mode"]}')
print('Model init: done.')

if config['util_mode'] == 'training':
    criterion = Flow_Mixture_Loss(**config).cuda()
else:
    criterion = None
print('Loss init: done.')

path2checkpoint = os.path.join(config['experiment_path'], config['model_name'])
checkpoint = torch.load(path2checkpoint, map_location='cpu')
model.load_state_dict(checkpoint['model_state'])
epoch = checkpoint['epoch']
print("epoch: ", epoch)
del(checkpoint)
print('Model {} loaded.'.format(path2checkpoint))

'''
if args.sampling:
    sample(eval_iterator, model, **config)
elif args.interpolation:
    interpolate(eval_iterator, model, **config)
'''
if config['util_mode'] == 'autoencoding' or config['util_mode'] == 'reconstruction':
    res = evaluate(eval_iterator, model, criterion, **config)
    print(res)
elif config['util_mode'] == 'generating':
    res = {}
    if config['jsd']:
        res['jsd'] = []
    if config['cd']:
        res['cd_mmds'], res['cd_covs'], res['cd_1nns'] = [], [], []
    if config['emd']:
        res['emd_mmds'], res['emd_covs'], res['emd_1nns'] = [], [], []
    if config['f1']:
        for _, f1_threshold in enumerate(config['f1_threshold_lst']):
            print('f1: ', f1_threshold)
            res['f1_%.4f_mmds'%(f1_threshold)] = []
            res['f1_%.4f_covs'%(f1_threshold)] = []
            res['f1_%.4f_1nns'%(f1_threshold)] = []
    for i in range(args.reps):
        res_per = evaluate(eval_iterator, model, criterion, **config)
        for key, value in enumerate(res_per):
            res[value].append(res_per[value])
    for key, value in enumerate(res):
        res[value] = torch.tensor(res[value], dtype=torch.float32)
        mean = torch.mean(res[value])
        std = torch.std(res[value])
        res[value] = '%.2f+-%.3f'%(mean.item(), std.item())
    print(res)
