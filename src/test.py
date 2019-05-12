import os
import argparse
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import common_init, bss_eval, soft_mask
from dataset import SC09MixtureDataset
from models import NVAE


class NVAETester:
    def __init__(self, sc09_mixture_root, partition, size, device):
        self.device = device
        self.partition = partition

        dataset = SC09MixtureDataset(root=sc09_mixture_root, partition=self.partition)
        if not size:
            size = len(dataset)
        dataloader = DataLoader(dataset, batch_size=size, shuffle=True)
        self.samples = next(iter(dataloader))
        self.samples['magnitude'] = self.samples['magnitude'].to(self.device)

    def run(self, checkpoints, step=None):
        start = time()

        train_base = os.path.join(checkpoints, 'train')

        if step:
            path = os.path.join(train_base, '{:06d}.pth'.format(step))
            nvae = NVAE.load(path=path, device=self.device)
            nvae.forward(x_mix=self.samples['magnitude'],
                         x_src=None,
                         y_sparse=self.samples['labels'],
                         train=False)
            p_x_src = nvae.batch_p_x_src
            filename = os.path.basename(path)
        else:
            p_x_src = SC09MixtureDataset.sources_to_magnitudes(self.samples['sources']).to(device=self.device)
            filename = 'oracle.pth'

        final_magnitudes, _ = soft_mask(p_x_src, self.samples['magnitude'])

        ref_waves = SC09MixtureDataset.sources_to_waves(self.samples['sources']).numpy()
        pred_waves = SC09MixtureDataset.magnitudes_to_waves(final_magnitudes.detach().cpu().numpy(), self.samples['phase'])
        results = bss_eval(ref_waves, pred_waves)

        sdr = np.median(results[:, 0, :]).item()
        sir = np.median(results[:, 1, :]).item()
        sar = np.median(results[:, 2, :]).item()

        print('[%s] elapsed: %.3f sdr: %.3f sir: %.3f sar: %.3f' % (
            filename,
            time()-start,
            sdr,
            sir,
            sar))

        test_base = os.path.join(checkpoints, 'test', self.partition)
        os.makedirs(test_base, exist_ok=True)
        torch.save(results, os.path.join(test_base, filename))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sc09-mixture-root', type=str, required=True)
    parser.add_argument('--checkpoints', type=str, required=True)
    parser.add_argument('--partition', type=str, default='testing')
    parser.add_argument('--size', type=int)
    parser.add_argument('--step', type=int)
    parser.add_argument('--device', type=str)
    args = parser.parse_args()

    if not args.device:
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    common_init()

    nvae_tester = NVAETester(sc09_mixture_root=args.sc09_mixture_root,
                             partition=args.partition,
                             size=args.size,
                             device=args.device)
    nvae_tester.run(checkpoints=args.checkpoints,
                    step=args.step)


if __name__ == '__main__':
    main()
