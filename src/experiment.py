import os
import argparse

import torch

from mix import SC09Mixer, TargetRootExistException
from train import NVAETrainer
from test import NVAETester
from utils import common_init


class Experimenter:
    def __init__(self, sc_root, experiments_root, experiment_name, device):
        self.mixer = SC09Mixer(sc_root)
        self.datasets_root = os.path.join(experiments_root, 'datasets')
        self.checkpoints_root = os.path.join(experiments_root, experiment_name)
        self.device = device

    def compare_ae_vae_source_label(self):
        for ae in (True, False):
            for supervision in ('source', 'label'):
                print('HYPERPARAMETERS', 'ae=%s supervision=%s' % (ae, supervision))

                dataset_root, dataset_name = self._create_dataset(class_start=0,
                                                                  class_count=10,
                                                                  num_components=2,
                                                                  num_training=15000)

                checkpoints, best_step = self._train(dataset_root=dataset_root,
                                                     dataset_name=dataset_name,
                                                     supervision=supervision,
                                                     batch_size=100,
                                                     latent_size=128,
                                                     num_filters=128,
                                                     beta=10,
                                                     ae=ae)

                self._test(dataset_root, checkpoints, best_step)

                print('*' * 100)

    def vary_num_classes(self):
        for class_count in range(3, 11):
            if class_count == 3:
                class_start_values = [0, 3, 6]
            else:
                class_start_values = [0]

            for class_start in class_start_values:
                print('HYPERPARAMETERS', 'class_start=%d class_count=%d' % (class_start, class_count))

                dataset_root, dataset_name = self._create_dataset(class_start=class_start,
                                                                  class_count=class_count,
                                                                  num_components=2,
                                                                  num_training=15000)

                checkpoints, best_step = self._train(dataset_root=dataset_root,
                                                     dataset_name=dataset_name,
                                                     supervision='label',
                                                     batch_size=100,
                                                     latent_size=128,
                                                     num_filters=128,
                                                     beta=10,
                                                     ae=False)

                self._test(dataset_root, checkpoints, best_step)

                print('*' * 100)

    def _create_dataset(self, class_start, class_count, num_components, num_training):
        dataset_name = 'sc09mix-cl%d_%d-cm%d-nt%d' % (class_count, class_start, num_components, num_training)
        dataset_root = os.path.join(self.datasets_root, dataset_name)
        print('creating dataset %s' % dataset_name)
        try:
            self.mixer.run(target_root=dataset_root,
                           classes=list(range(class_start, class_start + class_count)),
                           num_components=num_components,
                           num_training=num_training)
        except TargetRootExistException:
            print('dataset %s exists, skipped.' % dataset_name)
        return dataset_root, dataset_name

    def _train(self, dataset_root, dataset_name, supervision, batch_size, latent_size, num_filters,
               beta, ae):
        print('starting training')
        training_name = '%s-%s-bs%d-ls%d-hs%d-b%d-%s' % (
            dataset_name, supervision, batch_size, latent_size, num_filters, beta, 'ae' if ae else 'vae')
        checkpoints = os.path.join(self.checkpoints_root, training_name)

        nvae_trainer = NVAETrainer(sc09_mixture_root=dataset_root, device=self.device)
        best_step = nvae_trainer.run(checkpoints=checkpoints,
                                     supervision=supervision,
                                     batch_size=batch_size,
                                     latent_size=latent_size,
                                     num_filters=num_filters,
                                     beta=beta,
                                     ae=ae,
                                     report_interval=200,
                                     patience=10)
        print('best validation step: %d' % best_step)
        return checkpoints, best_step

    def _test(self, dataset_root, checkpoints, best_step):
        print('starting testing')
        nvae_tester = NVAETester(sc09_mixture_root=dataset_root,
                                 partition='testing',
                                 size=None,
                                 device=self.device)
        nvae_tester.run(checkpoints=checkpoints,
                        step=best_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sc-root', type=str, required=True)
    parser.add_argument('--experiments-root', type=str, required=True)
    parser.add_argument('--run', type=str, required=True)
    parser.add_argument('--device', type=str)
    args = parser.parse_args()

    if not args.device:
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    common_init()

    experimenter = Experimenter(sc_root=args.sc_root,
                                experiments_root=args.experiments_root,
                                experiment_name=args.run,
                                device=args.device)

    if args.run == 'compare_ae_vae_source_label':
        experimenter.compare_ae_vae_source_label()
    elif args.run == 'vary_num_classes':
        experimenter.vary_num_classes()
    else:
        raise Exception('invalid experiment')


if __name__ == '__main__':
    main()
