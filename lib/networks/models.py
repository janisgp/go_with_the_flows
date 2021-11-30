import torch
import torch.nn as nn

from .resnet import resnet18

from .encoders import PointNetCloudEncoder
from .encoders import FeatureEncoder

from .decoders import GlobalRNVPDecoder
from .decoders import LocalCondRNVPDecoder


class Local_Cond_RNVP_MC_Global_RNVP_VAE(nn.Module):
    '''
    Basic flow module from DPF: https://github.com/Regenerator/dpf-nets.git
    Separate to two parts: prior flow and decoder flow

    Args:
        train_mode: p_rnvp_mc_g_rnvp_vae (for generation and auto-encoding task) / p_rnvp_mc_g_rnvp_vae_ic (for reconstruction task)
        mode: train/generation/auto-encoding
        deterministic: true/false
        pc_*: configs for point cloud encoder
        g_*: configs for prior flow
        p_*: configs for decoder flow
        pc_encoder: point cloud encoder
        g0_prior_mus, g0_prior_logvars: prior distribution to be optimized
        g_prior: prior flow model
        g_posterior: encoder to extract distribution from input point cloud
        p_prior: encoder to extract distribution from input point cloud for decoder flow
        p_decoder_base_type: free/freevar/fixed
                             free: generation task
                             freevar: auto-encoding /svr task
                             fixed: distribution is fixed
        pc_decoder: decoder flow
    '''

    def __init__(self, **kwargs):
        super(Local_Cond_RNVP_MC_Global_RNVP_VAE, self).__init__()
        self.train_mode = kwargs.get('train_mode')
        self.mode = kwargs.get('util_mode')
        self.deterministic = kwargs.get('deterministic')

        self.pc_enc_init_n_channels = kwargs.get('pc_enc_init_n_channels')
        self.pc_enc_init_n_features = kwargs.get('pc_enc_init_n_features')
        self.pc_enc_n_features = kwargs.get('pc_enc_n_features')

        self.g_latent_space_size = kwargs.get('g_latent_space_size')

        self.g_prior_n_flows = kwargs.get('g_prior_n_flows')
        self.g_prior_n_features = kwargs.get('g_prior_n_features')

        self.g_posterior_n_layers = kwargs.get('g_posterior_n_layers')

        self.p_latent_space_size = kwargs.get('p_latent_space_size')
        self.p_prior_n_layers = kwargs.get('p_prior_n_layers')

        self.p_decoder_n_flows = kwargs.get('p_decoder_n_flows')
        self.p_decoder_n_features = kwargs.get('p_decoder_n_features')
        self.p_decoder_base_type = kwargs.get('p_decoder_base_type')
        self.p_decoder_base_var = kwargs.get('p_decoder_base_var')

        self.pc_encoder = PointNetCloudEncoder(self.pc_enc_init_n_channels,
                                               self.pc_enc_init_n_features,
                                               self.pc_enc_n_features)

        self.g0_prior_mus = nn.Parameter(torch.Tensor(1, self.g_latent_space_size))
        self.g0_prior_logvars = nn.Parameter(torch.Tensor(1, self.g_latent_space_size))
        with torch.no_grad():
            nn.init.normal_(self.g0_prior_mus.data, mean=0.0, std=0.033)
            nn.init.normal_(self.g0_prior_logvars.data, mean=0.0, std=0.33)

        self.g_prior = GlobalRNVPDecoder(self.g_prior_n_flows, self.g_prior_n_features,
                                         self.g_latent_space_size, weight_std=0.01)

        self.g_posterior = FeatureEncoder(self.g_posterior_n_layers, self.pc_enc_n_features[-1],
                                          self.g_latent_space_size, deterministic=False,
                                          mu_weight_std=0.0033, mu_bias=0.0,
                                          logvar_weight_std=0.033, logvar_bias=0.0)

        if self.p_decoder_base_type == 'free':
            self.p_prior = FeatureEncoder(self.p_prior_n_layers, self.g_latent_space_size,
                                          self.p_latent_space_size, deterministic=False,
                                          mu_weight_std=0.001, mu_bias=0.0,
                                          logvar_weight_std=0.01, logvar_bias=0.0)
        elif self.p_decoder_base_type == 'freevar':
            self.register_buffer('p_prior_mus', torch.zeros((1, self.p_latent_space_size, 1)))
            self.p_prior = FeatureEncoder(self.p_prior_n_layers, self.g_latent_space_size,
                                          self.p_latent_space_size, deterministic=True,
                                          mu_weight_std=0.01, mu_bias=0.0)
        elif self.p_decoder_base_type == 'fixed':
            self.register_buffer('p_prior_mus', torch.zeros((1, self.p_latent_space_size, 1)))
            self.register_buffer('p_prior_logvar', self.p_decoder_base_var * torch.ones((1, self.p_latent_space_size, 1)))

        self.pc_decoder = LocalCondRNVPDecoder(self.p_decoder_n_flows,
                                               self.p_decoder_n_features,
                                               self.g_latent_space_size,
                                               weight_std=0.01)

    def reparameterize(self, mu, logvar):
        '''
        function to reparamaterize as gaussian distribution ~ N(mu, exp(0.5 *logvar))

        Args:
            mu: mean
            logvar: log variance
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, g_input):
        '''
        function to encode prior flow

        Args:
            g_input: input point cloud

        Returns:
            output: g_prior_samples: output prior shape distribution list
                    g_prior_mus: output prior shape distribution of mus list
                    g_prior_logvars: output prior shape log deviations list
        '''
        output = {}
        output['g_prior_mus'] = [self.g0_prior_mus.expand(g_input.shape[0], self.g_latent_space_size)]
        output['g_prior_logvars'] = [self.g0_prior_logvars.expand(g_input.shape[0], self.g_latent_space_size)]
        if self.mode == 'training' or self.mode == 'autoencoding':
            p_enc_features = self.pc_encoder(g_input)
            g_enc_features = torch.max(p_enc_features, dim=2)[0]

            # get posterior distribution from point cloud features
            output['g_posterior_mus'], output['g_posterior_logvars'] = self.g_posterior(g_enc_features)
            output['g_posterior_samples'] = self.reparameterize(output['g_posterior_mus'],
                    output['g_posterior_logvars']) if self.mode == 'training' else output['g_posterior_mus']

            # train prior flow / auto-encoding task get prior distribution
            # g_prior_samples represents list of output transformations after couping layers
            buf_g = self.g_prior(output['g_posterior_samples'], mode='inverse')
            # inverse training, the last layer is the g_posterior_samples computed from the input
            # point cloud, used for loss computation.
            output['g_prior_samples'] = buf_g[0] + [output['g_posterior_samples']]
        elif self.mode == 'generating':
            # generation task, get prior distribution
            output['g_prior_samples'] = [self.reparameterize(output['g_prior_mus'][0], output['g_prior_logvars'][0])]
            buf_g = self.g_prior(output['g_prior_samples'][0], mode='direct')
            # direct transformation, the last layer is the predicted sample distribution
            output['g_prior_samples'] += buf_g[0]
        # g_prior_logvars returns the list of prior logvars generated after coupling layers
        # g_prior_mus returns the list of prior mus generated after coupling layers
        output['g_prior_mus'] += buf_g[1]
        output['g_prior_logvars'] += buf_g[2]
        return output

    def one_flow_decode(self, p_input, g_sample, pc_decoder, n_sampled_points):
        '''
        decode flow for one flow only

        Args:
            p_input: input point cloud
            g_sample: another input point cloud resampled from the same point cloud like p_input
            pc_decoder: decoder flow
            n_sampled_points: number of points need to be sampled

        Returns:
            output: p_prior_samples: the output decoder flow samples list
                    p_prior_mus: the output decoder flow mus list
                    p_prior_logvars: the output decoder flow log deviations list
        '''
        output = {}
        if self.p_decoder_base_type == 'free':
            # for training/generation task
            output['p_prior_mus'], output['p_prior_logvars'] = self.p_prior(g_sample)
            output['p_prior_mus'] = [output['p_prior_mus'].unsqueeze(2).expand(
                g_sample.shape[0], self.p_latent_space_size, n_sampled_points
            )]
            output['p_prior_logvars'] = [output['p_prior_logvars'].unsqueeze(2).expand(
                g_sample.shape[0], self.p_latent_space_size, n_sampled_points
            )]
        elif self.p_decoder_base_type == 'freevar':
            # for autoencoding/reconstruction task
            output['p_prior_mus'] = [self.p_prior_mus.expand(
                g_sample.shape[0], self.p_latent_space_size, n_sampled_points
            )]
            output['p_prior_logvars'] = [self.p_prior(g_sample).unsqueeze(2).expand(
                g_sample.shape[0], self.p_latent_space_size, n_sampled_points
            )]

        elif self.p_decoder_base_type == 'fixed':
            output['p_prior_mus'] = [self.p_prior_mus.expand(
                g_sample.shape[0], self.p_latent_space_size, n_sampled_points
            )]
            output['p_prior_logvars'] = [self.p_prior_logvar.expand(
                g_sample.shape[0], self.p_latent_space_size, n_sampled_points
            )]

        if self.mode == 'training':
            #train decoder flow
            buf = pc_decoder(p_input, g_sample, mode='inverse')
            output['p_prior_samples'] = buf[0] + [p_input]
        else:
            # for evaluation
            output['p_prior_samples'] = [self.reparameterize(output['p_prior_mus'][0], output['p_prior_logvars'][0])]
            buf = pc_decoder(output['p_prior_samples'][0], g_sample, mode='direct')
            output['p_prior_samples'] += buf[0]
        output['p_prior_mus'] += buf[1]
        output['p_prior_logvars'] += buf[2]

        return output

    '''
    def decode(self, p_input, g_sample, n_sampled_points):
    
        function to decode flow, for compatible with DPF that only uses one flow

         Args:
            p_input: input point cloud
            g_sample: another input point cloud resampled from the same point cloud like p_input
            n_sampled_points: number of points need to be sampled

        Returns:
            one_flow_decode

        return self.one_flow_decode(p_input, g_sample, self.pc_decoder, n_sampled_points)
    '''
    def forward(self, g_input, p_input, images=None, n_sampled_points=None, labeled_samples=False, warmup=False):
        '''
        main function

        Args:
            g_input: input point cloud B * 3 * N
            p_input: another input point cloud resampled from the same point cloud like p_input
            images: used for svr task
            n_sampled_points: number of points sampled, when training ,set to None
                              when evaluation, for generation/auto-encoding: 2048
                              for svr: 2500
            labeled_samples: true/false, output points labels or not.
            warmup: true/false, use warmup or not.

        Returns:
            output_encoder: shape distribution list after prior flow
            output_decoder: samples list after decoder flow
            mixture_weights_logits: log weight of each flow.
        '''

        sampled_cloud_size = p_input.shape[2] if n_sampled_points is None else n_sampled_points
        if images is not None and self.train_mode == 'p_rnvp_mc_g_rnvp_vae_ic':
            #for svr task
            output_encoder = self.encode(g_input, images)
        else:
            #for generation/auto-encoding task
            output_encoder = self.encode(g_input)
        g_sample = output_encoder['g_posterior_samples'] if self.mode == 'training' or self.mode == 'autoencoding' \
            else output_encoder['g_prior_samples'][-1]
        if labeled_samples:
            samples, labels, mixture_weights_logits = self.decode(p_input, g_sample, sampled_cloud_size, labeled_samples, warmup)
            return output_encoder, samples, labels, mixture_weights_logits
        else:
            output_decoder, mixture_weights_logits = self.decode(p_input, g_sample, sampled_cloud_size, labeled_samples, warmup)
            return output_encoder, output_decoder, mixture_weights_logits
