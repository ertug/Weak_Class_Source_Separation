import os
import argparse
from time import time

import torch
from torch.utils.data import DataLoader

from utils import common_init, total_params
from dataset import SC09MixtureDataset
from models import NVAE


class NVAETrainer:
    def __init__(self, sc09_mixture_root, device):
        self.train_dataset = SC09MixtureDataset(root=sc09_mixture_root, partition='training')
        self.num_sources = len(self.train_dataset[0]['sources'])

        train_eval_dataset = SC09MixtureDataset(root=sc09_mixture_root, partition='training')
        valid_eval_dataset = SC09MixtureDataset(root=sc09_mixture_root, partition='validation')
        eval_dataset_size = 1000  # len(valid_eval_dataset)
        self.train_eval_samples = self._load_eval_samples(device, dataset=train_eval_dataset, size=eval_dataset_size)
        self.valid_eval_samples = self._load_eval_samples(device, dataset=valid_eval_dataset, size=eval_dataset_size)
        print('using %d samples for evaluation' % eval_dataset_size)

        self.device = device

    def run(self, checkpoints, supervision, batch_size, latent_size, num_filters, beta, ae,
            report_interval, patience):

        os.makedirs(checkpoints, exist_ok=True)
        if os.listdir(checkpoints):
            raise Exception('checkpoints dir "%s" is not empty!' % checkpoints)

        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True)

        self.nvae = NVAE(device=self.device,
                         num_filters=num_filters,
                         latent_size=latent_size,
                         num_classes=SC09MixtureDataset.NUM_CLASSES,
                         num_sources=self.num_sources,
                         supervision=supervision,
                         reparameterize=(not ae))

        print('parameter count: q=%d p=%d' % (total_params(self.nvae.net_q_params), total_params(self.nvae.net_p_params)))

        self.stop = False
        self.cur_epoch = 0
        self.cur_batch = 0
        self.cur_patience = 0
        self.total_steps = 0
        self.report_start = time()
        self.last_best = {}
        self.best_step = 0

        self.optimizer = torch.optim.Adam([
            {'params': self.nvae.net_q_params},
            {'params': self.nvae.net_p_params},
        ], lr=1e-3, betas=(0.9, 0.999))

        while not self.stop:
            self.cur_batch = 0
            for samples in self.train_dataloader:
                if supervision == 'source':
                    x_src = SC09MixtureDataset.sources_to_magnitudes(samples['sources'])
                    x_loss_wrt = 'src'
                elif supervision == 'label':
                    x_src = None
                    x_loss_wrt = 'mix'
                else:
                    raise Exception('unknown supervision:', supervision)

                self.nvae.forward(x_mix=samples['magnitude'],
                                  x_src=x_src,
                                  y_sparse=samples['labels'],
                                  train=True)

                criterion = self._z_loss(beta, ae) + self._x_loss(x_loss_wrt)

                self.optimizer.zero_grad()
                criterion.backward()
                self.optimizer.step()

                print('.', end='', flush=True)
                if (self.total_steps+1) % report_interval == 0:
                    self._report(checkpoints, supervision, patience)
                if self.stop:
                    break

                self.total_steps += 1
                self.cur_batch += 1
            self.cur_epoch += 1

        return self.best_step

    def _report(self, checkpoints, supervision, patience):
        train_x_loss_mix, train_x_loss_src, train_x_loss_src_masked = self._eval(self.train_eval_samples)
        valid_x_loss_mix, valid_x_loss_src, valid_x_loss_src_masked = self._eval(self.valid_eval_samples)

        best_mix = self._is_best('mix', valid_x_loss_mix)
        best_src = self._is_best('src', valid_x_loss_src)
        best_src_masked = self._is_best('src_masked', valid_x_loss_src_masked)

        if supervision == 'source':
            is_best_step = best_src
        elif supervision in ('label', 'none'):
            is_best_step = best_mix
        else:
            raise Exception('unknown supervision:', supervision)

        if is_best_step:
            self.cur_patience = 0
            self.best_step = self.total_steps
            path = os.path.join(checkpoints, 'train', '{:06d}.pth'.format(self.total_steps))
            self.nvae.save(path)
        else:
            self.cur_patience += 1
        self.stop = self.cur_patience >= patience

        report_elapsed = time() - self.report_start
        self.report_start = time()

        print('\n[%6d, %3d] elapsed: %.3f mix: %.3f/%.3f %s src: %.3f/%.3f %s src_masked: %.3f/%.3f %s' % (
            self.total_steps,
            self.cur_epoch,
            report_elapsed,
            train_x_loss_mix,
            valid_x_loss_mix,
            '(B)' if best_mix else '   ',
            train_x_loss_src,
            valid_x_loss_src,
            '(B)' if best_src else '   ',
            train_x_loss_src_masked,
            valid_x_loss_src_masked,
            '(B)' if best_src_masked else ''))

    def _is_best(self, name, loss):
        if name not in self.last_best or loss <= self.last_best[name]:
            self.last_best[name] = loss
            return True
        else:
            return False

    def _z_loss(self, beta, ae):
        if ae:
            return 0
        else:
            pointwise_kl = -0.5 * (1 + self.nvae.batch_q_logvar - self.nvae.batch_q_mu.pow(2) -
                                   self.nvae.batch_q_logvar.exp())
            return torch.mean(torch.sum(pointwise_kl, dim=2)) * beta

    def _x_loss(self, wrt):
        if wrt == 'src':
            batch_x = self.nvae.batch_x_src
            batch_p_x = self.nvae.batch_p_x_src
        elif wrt == 'src_masked':
            batch_x = self.nvae.batch_x_src
            batch_p_x, _ = self.nvae.batch_p_x_src_masked()
        elif wrt == 'mix':
            batch_x = self.nvae.batch_x_mix
            batch_p_x = self.nvae.batch_p_x_src.sum(dim=1)
        else:
            raise Exception('unknown wrt:', wrt)

        pointwise_loss = batch_p_x - batch_x * batch_p_x.log()

        if wrt in ('src', 'src_masked'):
            return torch.mean(torch.sum(pointwise_loss, dim=(2, 3)))
        elif wrt == 'mix':
            return torch.mean(torch.sum(pointwise_loss, dim=(1, 2)))
        else:
            raise Exception('unknown wrt:', wrt)

    def _load_eval_samples(self, device, dataset, size):
        dataloader = DataLoader(dataset, batch_size=size, shuffle=True)
        samples = next(iter(dataloader))
        samples['magnitude'] = samples['magnitude'].to(device)
        return samples

    def _eval(self, samples):
        self.nvae.forward(x_mix=samples['magnitude'],
                          x_src=SC09MixtureDataset.sources_to_magnitudes(samples['sources']),
                          y_sparse=samples['labels'],
                          train=False)

        x_loss_mix = self._x_loss(wrt='mix')
        x_loss_src = self._x_loss(wrt='src')
        x_loss_src_masked = self._x_loss(wrt='src_masked')
        return x_loss_mix, x_loss_src, x_loss_src_masked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sc09-mixture-root', type=str, required=True)
    parser.add_argument('--checkpoints', type=str, required=True)
    parser.add_argument('--supervision', choices=['label', 'source'], default='label')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--latent-size', type=int, default=128)
    parser.add_argument('--num-filters', type=int, default=128)
    parser.add_argument('--beta', type=float, default=10)
    parser.add_argument('--ae', action='store_true')
    parser.add_argument('--report-interval', type=int, default=200)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str)
    args = parser.parse_args()

    if not args.device:
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    common_init()

    nvae_trainer = NVAETrainer(args.sc09_mixture_root, args.device)
    nvae_trainer.run(checkpoints=args.checkpoints,
                     supervision=args.supervision,
                     batch_size=args.batch_size,
                     latent_size=args.latent_size,
                     num_filters=args.num_filters,
                     beta=args.beta,
                     ae=args.ae,
                     report_interval=args.report_interval,
                     patience=args.patience)


if __name__ == '__main__':
    main()
