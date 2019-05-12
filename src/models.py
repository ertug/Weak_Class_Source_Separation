import os

import torch
from torch import nn
import torch.nn.functional as F

from dataset import SC09MixtureDataset
from utils import soft_mask


def init_weights(w):
    if isinstance(w, nn.Conv2d):
        nn.init.kaiming_normal_(w.weight.data, mode='fan_in', nonlinearity='relu')
        w.bias.data.zero_()
    elif isinstance(w, nn.Linear):
        nn.init.kaiming_normal_(w.weight.data, mode='fan_in', nonlinearity='relu')
        w.bias.data.zero_()


class SC09Encoder(nn.Module):
    def __init__(self, nf, nz, t, f):
        super(SC09Encoder, self).__init__()
        self.nf = nf
        self.t = t
        self.f = f

        self.conv_1 = nn.Conv2d(1, self.nf, (1, self.f), (1, 1), (0, 0))
        self.batch_norm_1 = nn.BatchNorm2d(self.nf)
        self.conv_2 = nn.Conv2d(self.nf, self.nf, (4, 1), (2, 1), (1, 0))
        self.batch_norm_2 = nn.BatchNorm2d(self.nf)
        self.conv_3 = nn.Conv2d(self.nf, self.nf * 2, (4, 1), (2, 1), (1, 0))
        self.batch_norm_3 = nn.BatchNorm2d(self.nf * 2)
        self.linear_4 = nn.Linear(self.nf * 2 * self.t//4, self.nf * 4)
        self.batch_norm_4 = nn.BatchNorm1d(self.nf * 4)
        self.linear_51 = nn.Linear(self.nf * 4, nz)
        self.linear_52 = nn.Linear(self.nf * 4, nz)

        self.apply(init_weights)

    def forward(self, x):
        h = x.view(-1, 1, self.t, self.f)
        h = F.relu(self.batch_norm_1(self.conv_1(h)))
        h = F.relu(self.batch_norm_2(self.conv_2(h)))
        h = F.relu(self.batch_norm_3(self.conv_3(h)))
        h = h.view(-1, self.nf * 2 * self.t//4)
        h = F.relu(self.batch_norm_4(self.linear_4(h)))
        mu = self.linear_51(h)
        logvar = self.linear_52(h)
        return mu, logvar


class SC09Decoder(nn.Module):
    def __init__(self, nf, nz, t, f):
        super(SC09Decoder, self).__init__()
        self.nf = nf
        self.t = t
        self.f = f

        self.linear_1 = nn.Linear(nz, self.nf * 2 * self.t//4)
        self.batch_norm_1 = nn.BatchNorm2d(self.nf * 2)
        self.conv_trans_2 = nn.ConvTranspose2d(self.nf * 2, self.nf, (4, 1), (2, 1), (1, 0))
        self.batch_norm_2 = nn.BatchNorm2d(self.nf)
        self.conv_trans_3 = nn.ConvTranspose2d(self.nf, self.nf, (4, 1), (2, 1), (1, 0))
        self.batch_norm_3 = nn.BatchNorm2d(self.nf)
        self.conv_trans_4 = nn.ConvTranspose2d(self.nf, 1, (1, self.f), (1, 1), (0, 0))

        self.apply(init_weights)

    def forward(self, z):
        h = self.linear_1(z).view(-1, self.nf * 2, self.t//4, 1)
        h = F.relu(self.batch_norm_1(h))
        h = F.relu(self.batch_norm_2(self.conv_trans_2(h)))
        h = F.relu(self.batch_norm_3(self.conv_trans_3(h)))
        x = self.conv_trans_4(h)
        return F.softplus(x)


class NVAE:
    def __init__(self, device, num_filters, latent_size, num_classes, num_sources, supervision,
                 reparameterize=True):
        self.device = device
        self.num_filters = num_filters
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.num_sources = num_sources
        self.supervision = supervision
        self.reparameterize = reparameterize

        self.net_q_list = [self._create_encoder() for _ in range(self.num_classes)]
        self.net_p_list = [self._create_decoder() for _ in range(self.num_classes)]

        self.net_q_params = sum([list(net_q.parameters()) for net_q in self.net_q_list], [])
        self.net_p_params = sum([list(net_p.parameters()) for net_p in self.net_p_list], [])

        self.batch_x_mix = torch.empty(0, device=self.device)
        self.batch_x_src = torch.empty(0, device=self.device)
        self.batch_y_sparse = None
        self.batch_q_mu = torch.empty(0, device=self.device)
        self.batch_q_logvar = torch.empty(0, device=self.device)
        self.batch_p_x_src = torch.empty(0, device=self.device)

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path)

        instance = cls(device=device,
                       num_filters=checkpoint['num_filters'],
                       latent_size=checkpoint['latent_size'],
                       num_classes=checkpoint['num_classes'],
                       num_sources=checkpoint['num_sources'],
                       supervision=checkpoint['supervision'],
                       reparameterize=checkpoint['reparameterize'])

        for i, net_q in enumerate(instance.net_q_list):
            instance._load_single(checkpoint['net_q_state_dict_list'][i], net_q)
        for i, net_p in enumerate(instance.net_p_list):
            instance._load_single(checkpoint['net_p_state_dict_list'][i], net_p)

        return instance

    @staticmethod
    def _load_single(state_dict, net):
        net.load_state_dict(state_dict)
        for param in net.parameters():
            param.requires_grad = False
        net.eval()

    def save(self, path):
        checkpoint = {
            'num_filters': self.num_filters,
            'latent_size': self.latent_size,
            'num_classes': self.num_classes,
            'num_sources': self.num_sources,
            'supervision': self.supervision,
            'reparameterize': self.reparameterize,
            'net_q_state_dict_list': [net_q.state_dict() for net_q in self.net_q_list],
            'net_p_state_dict_list': [net_p.state_dict() for net_p in self.net_p_list],
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)

    def forward(self, x_mix, x_src, y_sparse, train):
        current_batch_size = x_mix.size(0)

        self.batch_x_mix.resize_(current_batch_size, SC09MixtureDataset.CONTEXT_FRAMES, SC09MixtureDataset.SPECTROGRAM_BINS)
        self.batch_x_mix.copy_(x_mix)
        if x_src is None:
            self.batch_x_src.zero_()
        else:
            self.batch_x_src.resize_(current_batch_size, self.num_sources,
                                     SC09MixtureDataset.CONTEXT_FRAMES, SC09MixtureDataset.SPECTROGRAM_BINS)
            self.batch_x_src.copy_(x_src)
        self.batch_y_sparse = y_sparse

        for net_q in self.net_q_list:
            net_q.train(train)
        for net_p in self.net_p_list:
            net_p.train(train)

        self.batch_q_mu.detach_()
        self.batch_q_logvar.detach_()
        self.batch_p_x_src.detach_()

        self.batch_q_mu.resize_(current_batch_size, self.num_sources, self.latent_size).zero_()
        self.batch_q_logvar.resize_(current_batch_size, self.num_sources, self.latent_size).zero_()
        self.batch_p_x_src.resize_(current_batch_size, self.num_sources,
                                   SC09MixtureDataset.CONTEXT_FRAMES, SC09MixtureDataset.SPECTROGRAM_BINS).zero_()

        batch_by_labels = {}
        for batch_index, row in enumerate(self.batch_y_sparse):
            component_index = 0
            for label_tensor in row:
                label = label_tensor.item()
                if label != -1:
                    try:
                        batch = batch_by_labels[label]
                    except KeyError:
                        batch = batch_by_labels[label] = ([], [])
                    batch[0].append(batch_index)
                    batch[1].append(component_index)
                    component_index += 1

        for label, (batch_idx, component_idx) in batch_by_labels.items():
            # batch norm doesn't work with a single sample in the batch
            if train and len(batch_idx) <= 1:
                print('!!!single sample in batch!!!')
            else:
                q_mu, q_logvar, p_x_src = self._forward_single(label, self.batch_x_mix[batch_idx], train)
                self.batch_q_mu[batch_idx, component_idx] = q_mu
                self.batch_q_logvar[batch_idx, component_idx] = q_logvar
                self.batch_p_x_src[batch_idx, component_idx] = p_x_src.squeeze()

    def batch_p_x_src_masked(self):
        return soft_mask(self.batch_p_x_src, self.batch_x_mix)

    def _forward_single(self, label, x, train):
        q_mu, q_logvar = self.net_q_list[label](x)
        if train and self.reparameterize:
            z = self._reparameterize(q_mu, q_logvar)
        else:
            z = q_mu
        p_x_src = self.net_p_list[label](z)
        return q_mu, q_logvar, p_x_src

    @staticmethod
    def _reparameterize(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def _create_encoder(self):
        return SC09Encoder(nf=self.num_filters,
                           nz=self.latent_size,
                           t=SC09MixtureDataset.CONTEXT_FRAMES,
                           f=SC09MixtureDataset.SPECTROGRAM_BINS).to(self.device)

    def _create_decoder(self):
        return SC09Decoder(nf=self.num_filters,
                           nz=self.latent_size,
                           t=SC09MixtureDataset.CONTEXT_FRAMES,
                           f=SC09MixtureDataset.SPECTROGRAM_BINS).to(self.device)
