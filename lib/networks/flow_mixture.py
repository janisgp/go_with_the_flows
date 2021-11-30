import math
import numpy as np
import torch
import torch.nn as nn
from lib.networks.models import Local_Cond_RNVP_MC_Global_RNVP_VAE
from lib.networks.decoders import LocalCondRNVPDecoder
from lib.networks.resnet import resnet18
from lib.networks.encoders import FeatureEncoder, WeightsEncoder


class Flow_Mixture_Model(Local_Cond_RNVP_MC_Global_RNVP_VAE):
    ''' Train class for flow mixture model for generation and autoencoding
    Args:
        n_components: number of flos that are used
        params_reduce_mode: the way to choose  featrue size and number of coupling layers for each flow
                            4 types to choose: none, depth_first, feature_first, depth_and_feature
                                none: uses only one flow
                                depth_first: first satisfy more coupling layers
                                feature_first: first satisfy feature size
        weight_type: the way to generate weights of each flow
                     2 types to choose: global_weights: weights are defined using network parameters.
                                        learned_weights: weights are learn by WeightsEncoder
        mixture_weights_logits: the weight of each flow if using global weights
        pc_decoder: n_components of point cloud decoders
        mixture_weights_encoder: when using learned weights, it's used to generate weights for all flows
    '''
    def __init__(self, **kwargs):
        super(Flow_Mixture_Model, self).__init__(**kwargs)
        self.n_components = kwargs['n_components']
        self.params_reduce_mode = kwargs['params_reduce_mode']
        self.weights_type = kwargs['weights_type']
        self.mixture_weights_logits = torch.nn.Parameter(torch.zeros(self.n_components), requires_grad=True)
        p_decoder_n_flows, p_decoder_n_features = self._get_decoder_params()
        self.pc_decoder = nn.ModuleList([LocalCondRNVPDecoder(p_decoder_n_flows,
                                                              p_decoder_n_features,
                                                              self.g_latent_space_size,
                                                              weight_std=0.01) for _ in range(self.n_components)])
        
        self.mixture_weights_encoder = WeightsEncoder(3, self.g_latent_space_size,
                                          self.n_components, deterministic=True,
                                          mu_weight_std=0.001, mu_bias=0.0,
                                          logvar_weight_std=0.01, logvar_bias=0.0)
    
    def _get_decoder_params(self):
        '''
        according to different params reduce mode, decide feature size and number of coupling layers.
        Returns:
            decoder_depth: coupling layers in each decoder flow
            p_decoder_n_features: feature size in each decoder flow
        '''
        n = self.n_components
        if n == 1 or self.params_reduce_mode == 'none':  # if n == 1 or not reduction return standard params
            return self.p_decoder_n_flows, self.p_decoder_n_features
        else:  # if n != compute reduced params
            if self.params_reduce_mode == 'depth_and_feature':
                decoder_depth = math.ceil(self.p_decoder_n_flows / math.sqrt(n))
                p_decoder_n_features, _ = self._get_p_decoder_n_features(decoder_depth)
            elif self.params_reduce_mode == 'depth_first':
                # we ceil to ensure minimum depth 1 in regular cases
                decoder_depth = math.ceil(self.p_decoder_n_flows / n)
                p_decoder_n_features, _ = self._get_p_decoder_n_features(decoder_depth)
            elif self.params_reduce_mode == 'feature_first':
                decoder_depth = self.p_decoder_n_flows
                p_decoder_n_features, output = self._get_p_decoder_n_features(decoder_depth)
                if output[0]:
                    large_decoder_count = output[1]
                    current_total = output[2]
                    while current_total > large_decoder_count:  # lets keep at least 4 features :)
                        decoder_depth -= 1
                        current_decoder_count = LocalCondRNVPDecoder.get_param_count(decoder_depth,
                                                                                     p_decoder_n_features,
                                                                                     self.g_latent_space_size)
                        current_total = current_decoder_count * self.n_components
            else:
                raise ValueError(f'Unknown params_reduce_mode: {self.params_reduce_mode}')
            return decoder_depth, p_decoder_n_features

    def _get_p_decoder_n_features(self, depth):
        '''
        uses to decide the feature size.

        Args:
            depth: number of coupling layers

        Returns:
            p_decoder_n_features: feature size of each coupling layer
        '''
        p_decoder_n_features = self.p_decoder_n_features
        large_decoder_count = LocalCondRNVPDecoder.get_param_count(self.p_decoder_n_flows,
                                                                   p_decoder_n_features,
                                                                   self.g_latent_space_size)
        current_total = large_decoder_count * self.n_components
        while current_total > large_decoder_count and p_decoder_n_features > 4:  # lets keep at least 4 features :)
            p_decoder_n_features -= 1
            current_decoder_count = LocalCondRNVPDecoder.get_param_count(depth,
                                                                         p_decoder_n_features,
                                                                         self.g_latent_space_size)
            current_total = current_decoder_count * self.n_components
        return p_decoder_n_features, \
               (current_total > large_decoder_count,
                large_decoder_count,
                current_total)  # we need 2nd output for feature_first

    def get_weights(self, g_sample, warmup=False):
        '''
        decide the weights of all flows

        Args:
            g_sample: input point cloud
            warmup: if use warmup, then in the first few epochs, we use global weights type
                    else, we use learned weights type.
        Returns:
            mixture_weights_logits: log weights of each flow in decoder flows.
        '''
        if warmup or self.weights_type == 'global_weights':
            mixture_weights_logits = self.mixture_weights_logits.unsqueeze(0).expand(g_sample.shape[0], self.n_components)
        elif self.weights_type == 'learned_weights':
            mixture_weights_logits = self.mixture_weights_encoder(g_sample)
        
        return mixture_weights_logits

    def decode(self, p_input, g_sample, n_sampled_points, labeled_samples=False, warmup=False):
        '''
        mixtures of flows in decoder.

        Args:
            p_input: input point cloud  B * 3 * N
            g_sample: another sampled point cloud, from the same shape as p_input   B * 3 * N
            n_sampled_points: number of sampled points, when training,it's the number of points in p_input.
                              when evaluation, it's 2048 for generation /autoencoding, 2500 for svr.
            labeled_samples: if true, output labels (each point belongs to which flow), used in evaluation
                             if false, only output generated point cloud, and the mixtures weights
            warmup: if true, use global weights at first
                    else, use learned weights
        Returns:
            samples: output point clouds with labels
            labels: point labels
            mixture_weights_logits: weight of each flow
            output_decoder: output point clouds list
        '''
        mixture_weights_logits = self.get_weights(g_sample, warmup)
        if self.mode == 'training':
            sampled_cloud_size = [n_sampled_points for _ in range(self.n_components)]
        else:
            #when evaluation, each time, only one shape is inputed
            assert p_input.shape[0] == 1

            #computes the probabilities of all flows
            logits_exp = np.exp(mixture_weights_logits[0].detach().cpu().numpy())
            probs = logits_exp / logits_exp.sum()

            #for each flow, randomly choose certain number of points based on its probability
            flows_idx = np.random.choice(range(self.n_components), size=n_sampled_points, p=probs)

            #masks designs the labels
            masks = []
            for t in range(self.n_components):
                mask = flows_idx == t
                masks.append(mask.sum())
            sampled_cloud_size = masks

        output_decoder = []
        for i in range(self.n_components):
            #generate output parts for each flow decoder
            one_decoder = self.one_flow_decode(p_input, g_sample, self.pc_decoder[i], sampled_cloud_size[i])
            output_decoder.append(one_decoder)

        if labeled_samples:     #when for evaluation
            samples = torch.zeros_like((p_input))
            labels = torch.zeros(p_input.size(0), p_input.size(2))
            for t in range(self.n_components):
                #for each point, find its labels (generated by which flow)
                s = output_decoder[t]
                mask = flows_idx == t
                samples[:, :, mask] = s['p_prior_samples'][-1]
                labels[:, mask] = t + 1
            return samples, labels, mixture_weights_logits
        else:
            return output_decoder, mixture_weights_logits

class Flow_Mixture_SVR_Model(Flow_Mixture_Model):
    ''' Train class for flow mixture model for single view reconstruction
    Args:
        img_encoder: encoder used for encoding image
        g_prior_n_layers: the coupling layers for prior flow
        g0_prior: encoder used to extract mus and logvars from image features
    '''
    def __init__(self, **kwargs):
        super(Flow_Mixture_SVR_Model, self).__init__(**kwargs)
        self.img_encoder = resnet18(num_classes=self.g_latent_space_size)
        self.g_prior_n_layers = kwargs.get('g_prior_n_layers')
        self.g0_prior = FeatureEncoder(self.g_prior_n_layers, self.g_latent_space_size,
                                       self.g_latent_space_size, deterministic=False,
                                       mu_weight_std=0.0033, mu_bias=0.0,
                                       logvar_weight_std=0.033, logvar_bias=0.0)
        self.g0_prior_mus = None
        self.g0_prior_logvars = None
    def encode(self, g_input, images):
        '''
        encoder used to train prior flow.

        Args:
            g_input: input point cloud B * 3 * N
            images: input image  B * 4 * 224 * 224

        Returns:
            output: output shape distributions list after prior flow
        '''
        output = {}
        img_features = self.img_encoder(images)
        output['g_prior_mus'], output['g_prior_logvars'] = self.g0_prior(img_features)
        output['g_prior_mus'], output['g_prior_logvars'] = [output['g_prior_mus']], [output['g_prior_logvars']]
        if self.mode == 'training':
            #when training, get posterior from input point cloud
            p_enc_features = self.pc_encoder(g_input)
            g_enc_features = torch.max(p_enc_features, dim=2)[0]
            output['g_posterior_mus'], output['g_posterior_logvars'] = self.g_posterior(g_enc_features)
            output['g_posterior_samples'] = self.reparameterize(output['g_posterior_mus'], output['g_posterior_logvars'])

            # train the prior flow
            buf_g = self.g_prior(output['g_posterior_samples'], mode='inverse')
            output['g_prior_samples'] = buf_g[0] + [output['g_posterior_samples']]
        elif self.mode == 'reconstruction':
            #when reconstruction, use the trained prior flow to generate the prior from input image
            output['g_prior_samples'] = [output['g_prior_mus'][0]]
            buf_g = self.g_prior(output['g_prior_samples'][0], mode='direct')
            output['g_prior_samples'] += buf_g[0]
        output['g_prior_mus'] += buf_g[1]
        output['g_prior_logvars'] += buf_g[2]
        return output
