import numpy as np

import torch
import torch.nn as nn


class PointFlowNLL(nn.Module):
    def __init__(self):
        super(PointFlowNLL, self).__init__()

    def forward(self, output_decoder, **kwargs):
        cur_mus = output_decoder['p_prior_mus'][0]
        cur_logvars = output_decoder['p_prior_logvars'][0]
        cur_log_determinant = sum(output_decoder['p_prior_logvars'])
        cur_samples = output_decoder['p_prior_samples'][0]
        return 0.5 * torch.add(
            torch.sum(cur_log_determinant + ((cur_samples - cur_mus) ** 2 / torch.exp(cur_logvars)),
                      dim=1, keepdim=True),
            np.log(2.0 * np.pi) * cur_samples.shape[1]
        )


class GaussianFlowNLL(nn.Module):
    def __init__(self):
        super(GaussianFlowNLL, self).__init__()

    def forward(self, samples, mus, logvars):
        return 0.5 * torch.add(
            torch.sum(sum(logvars) + ((samples[0] - mus[0])**2 / torch.exp(logvars[0]))) / samples[0].shape[0],
            np.log(2.0 * np.pi) * samples[0].shape[1]
        )


class GaussianEntropy(nn.Module):
    def __init__(self):
        super(GaussianEntropy, self).__init__()

    def forward(self, logvars):
        return 0.5 * torch.add(logvars.shape[1] * (1.0 + np.log(2.0 * np.pi)), logvars.sum(1).mean())


class Local_Cond_RNVP_MC_Global_RNVP_VAE_Loss(nn.Module):
    def __init__(self, **kwargs):
        super(Local_Cond_RNVP_MC_Global_RNVP_VAE_Loss, self).__init__()
        self.pnll_weight = kwargs.get('pnll_weight')
        self.gnll_weight = kwargs.get('gnll_weight')
        self.gent_weight = kwargs.get('gent_weight')
        self.PNLL = PointFlowNLL()
        self.GNLL = GaussianFlowNLL()
        self.GENT = GaussianEntropy()

    def forward(self, g_clouds, l_clouds, outputs):
        pnll = torch.sum(self.PNLL(outputs['p_prior_samples'], outputs['p_prior_mus'], outputs['p_prior_logvars']))
        gnll = self.GNLL(outputs['g_prior_samples'], outputs['g_prior_mus'], outputs['g_prior_logvars'])
        gent = self.GENT(outputs['g_posterior_logvars'])
        return self.pnll_weight * pnll + self.gnll_weight * gnll - self.gent_weight * gent, pnll, gnll, gent

'''
class FlowMixtureNLL(PointFlowNLL):
    def __init__(self):
        super(FlowMixtureNLL, self).__init__()

    def forward(self, output_decoder, mixture_weights_logits=None):
        mixture_weights_norm = (torch.logsumexp(mixture_weights_logits, dim=-1)).unsqueeze(-1)
        weights_unnormed = torch.exp(mixture_weights_logits)
        log_weights = torch.log(weights_unnormed) - mixture_weights_norm
        log_weights = log_weights.unsqueeze(1)
        log_probs_components = []
        for i in range(len(output_decoder)):
            log_probs_components.append(-1 * super().forward(output_decoder=output_decoder[i]))
        print(log_probs_components[0].shape)
        log_probs_components = torch.transpose(torch.cat(log_probs_components, dim=1), 1, 2)
        print(log_probs_components.shape)
        log_probs_components = log_probs_components + log_weights
        log_probs_pnll = torch.logsumexp(log_probs_components, dim=-1)
        log_probs_nll = -1 * torch.mean(torch.sum(log_probs_pnll, dim=-1))

        return log_probs_nll
'''

class FlowMixtureNLL(nn.Module):
    '''
    class defines decoder flows loss
    '''
    def __init__(self):
        super(FlowMixtureNLL, self).__init__()

    def forward(self, output_decoder, mixture_weights_logits):
        '''
        main function for computing decoder flows loss

        Args:
            output_decoder: output samples list for decoder flows
            mixture_weights_logits: log weights for all flows in decoder flows

        Returns:
            loss for decoder flows
        '''

        # compute log weights
        mixture_weights_norm = (torch.logsumexp(mixture_weights_logits, dim=-1)).unsqueeze(-1)
        weights_unnormed = torch.exp(mixture_weights_logits)
        log_weights = torch.log(weights_unnormed) - mixture_weights_norm
        log_weights = log_weights.unsqueeze(1)
        
        num_patches = len(output_decoder)
        num_batches = output_decoder[0]['p_prior_mus'][0].shape[0]
        pnll = []
        for i in range(num_batches):
            loss_pnll_over_patches = []
            for j in range(num_patches):
                cur_mus = output_decoder[j]['p_prior_mus'][0][i, :, :]
                cur_logvars = output_decoder[j]['p_prior_logvars'][0][i, :, :]
                # compute sum of log determinant of each shape
                cur_log_determinant = sum(output_decoder[j]['p_prior_logvars'])[i, :, :]
                cur_samples = output_decoder[j]['p_prior_samples'][0][i, :, :]

                # compute the log probability of each shape in each flow
                part_1 = -torch.sum(cur_log_determinant + ((cur_samples - cur_mus) ** 2 / torch.exp(cur_logvars)),
                                    dim=0, keepdim=True)
                part_2 = -np.log(2.0 * np.pi) * cur_samples.shape[0]
                cur_pnll = 0.5 * torch.add(part_1, part_2)
                loss_pnll_over_patches.append(cur_pnll)
            loss_pnll_over_patches = torch.transpose(torch.cat(loss_pnll_over_patches, dim=0), 0, 1)

            # compute the log probability of each shape in all flows by adding its log weights
            log_probs_pnll = loss_pnll_over_patches + log_weights[i]
            log_probs_pnll = torch.logsumexp(log_probs_pnll, dim=-1)
            log_probs_pnll = -torch.sum(log_probs_pnll)

            pnll.append(log_probs_pnll.unsqueeze(0))

        # compute the average loss over the batch
        pnll = torch.cat(pnll)
        pnll = torch.mean(pnll)

        return pnll


class Flow_Mixture_Loss(nn.Module):
    '''
    class defines the loss function of flow mixture model

    Args:
        pnll_weight: 1, weight of decoder flows loss
        gnll_weight: 1, weight of prior flow loss
        gent_weight: entropy loss of posterior
    '''
    def __init__(self, **kwargs):
        super(Flow_Mixture_Loss, self).__init__()
        self.pnll_weight = kwargs.get('pnll_weight')
        self.gnll_weight = kwargs.get('gnll_weight')
        self.gent_weight = kwargs.get('gent_weight')
        self.n_components = kwargs.get('n_components')
        self.PNLL = FlowMixtureNLL()
        self.GNLL = GaussianFlowNLL()
        self.GENT = GaussianEntropy()

    def forward(self, output_prior, output_decoder, mixture_weights_logits):
        '''
        main function to compute losses for flow mixture model

        Args:
            output_prior: shape distributions list for prior flow
            output_decoder: samples list for decoder flows
            mixture_weights_logits: log weights of all flows in decoder flows
        Returns:
            sum of loss, decoder flow loss, gnll and gent used for KL divergence
        '''
        pnll = self.PNLL(output_decoder, mixture_weights_logits)
        gnll = self.GNLL(output_prior['g_prior_samples'], output_prior['g_prior_mus'], output_prior['g_prior_logvars'])
        gent = self.GENT(output_prior['g_posterior_logvars'])
        return self.pnll_weight * pnll + self.gnll_weight * gnll - self.gent_weight * gent, pnll, gnll, gent
