import argparse
import os
import io
import yaml

import torch
from torch.utils.data import DataLoader
from lib.datasets.datasets import ShapeNetCoreDataset
from lib.datasets.cloud_transformations import ComposeCloudTransformation
from lib.networks.flow_mixture import Flow_Mixture_Model
from lib.networks.training import predict
from lib.networks.utils import cnt_params


def define_options_parser():
    parser = argparse.ArgumentParser(description='Model training script. Provide a suitable config.')
    parser.add_argument('experiment_path', type=str, help='Path to experiment.')
    parser.add_argument('modelname', type=str, help='Model name to save checkpoints.')
    return parser


parser = define_options_parser()
args = parser.parse_args()
with io.open(os.path.join(args.experiment_path, 'config.yaml'), 'r') as stream:
    config = yaml.load(stream)
config['experiment_path'] = args.experiment_path
config['model_name'] = '{0}.pkl'.format(args.modelname)
config['util_mode'] = 'evaluating'
print('Configurations loaded.')

cloud_transform = ComposeCloudTransformation(**config)
test_dataset = ShapeNetCoreDataset(config['path2data'],
                                   part='val', meshes_fname=config['meshes_fname'],
                                   cloud_size=config['cloud_size'], return_eval_cloud=True,
                                   return_original_scale=config['cloud_rescale2orig'],
                                   cloud_transform=cloud_transform,
                                   chosen_label=config['chosen_label'])
print('Dataset init: done.')

test_iterator = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False,
                           num_workers=config['num_workers'], pin_memory=True, drop_last=True)
print('Iterator init: done.')

model = Flow_Mixture_Model(**config).cuda()
print('Model init: done.')
print('Total number of parameters: {}'.format(cnt_params(model.parameters())))
print('Total number of parameters: {}'.format(cnt_params(model.pc_decoder.parameters())))

path2checkpoint = os.path.join(config['experiment_path'], config['model_name'])
checkpoint = torch.load(path2checkpoint, map_location='cpu')
model.load_state_dict(checkpoint['model_state'])
del(checkpoint)
print('Model {} loaded.'.format(path2checkpoint))

predict(test_iterator, model, **config)
