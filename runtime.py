import argparse
import os
import io
import yaml
import time
import numpy as np
import pickle

import torch
from torch.utils.data import DataLoader
from lib.datasets.datasets import ShapeNetCoreDataset
from lib.datasets.cloud_transformations import ComposeCloudTransformation
from lib.networks.flow_mixture import Flow_Mixture_Model


def define_options_parser():
    parser = argparse.ArgumentParser(description='Runtime evaluation script.')
    parser.add_argument('config', type=str, help='Path of config.')
    return parser


parser = define_options_parser()
args = parser.parse_args()
with io.open(args.config, 'r') as stream:
    config = yaml.load(stream)
config['util_mode'] = 'generation'

cloud_transform = ComposeCloudTransformation(**config)
test_dataset = ShapeNetCoreDataset(config['path2data'],
                                    part='test', meshes_fname=config['meshes_fname'],
                                    cloud_size=config['cloud_size'], return_eval_cloud=True,
                                    return_original_scale=config['cloud_rescale2orig'],
                                    cloud_transform=cloud_transform,
                                    chosen_label=config['chosen_label'])
print('Dataset init: done.')
eval_iterator = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'],
                           num_workers=config['num_workers'], pin_memory=True, drop_last=True)
print('Iterator init: done.')

average_runtimes = []
number_of_components = np.arange(1, 11)
for n_components in number_of_components:

    config['n_components'] = n_components
    model = Flow_Mixture_Model(**config).cuda()
    print(f'Model with {model.n_components} components init: done.')

    model.eval()
    torch.set_grad_enabled(False)

    p_prior_samples, g_prior_samples = torch.ones((1, 3, 5000)).cuda(),\
                                       torch.ones((1, config['g_prior_n_features'])).cuda()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(1000):
            _ = model.pc_decoder[0](p_prior_samples, g_prior_samples, mode='direct')
    average_runtimes.append(time.time() - start_time)

pickle.dump([number_of_components, average_runtimes],
            open(os.path.join(config['path2save'], f'average_runtimes_{config["params_reduce_mode"]}.p'), 'wb'))

print('done')
