import os
import argparse
import torch
from torch.utils.data import DataLoader
from itertools import cycle, chain, combinations

from dataset import SC09Dataset, SC09MixtureDataset


class TargetRootExistException(Exception):
    pass


class SC09Mixer:
    TRAINING_RMS_MEAN = 0.067

    def __init__(self, sc_root):
        self.dataloaders = self._create_dataloaders(sc_root)

    def run(self, target_root, classes, num_components, num_training):
        os.makedirs(target_root, exist_ok=True)
        if os.listdir(target_root):
            raise TargetRootExistException('target "%s" is not empty!' % target_root)

        target_labels = self._generate_target_labels(classes, num_components)
        print('target labels:', target_labels)

        training_percentage = 100 - SC09Dataset.VALIDATION_PERCENTAGE - SC09Dataset.TESTING_PERCENTAGE

        print('building training set...')
        self._create_dataset(partition='training',
                             target_labels=target_labels,
                             count=num_training,
                             target_root=target_root,
                             num_components=num_components)

        print('building validation set...')
        self._create_dataset(partition='validation',
                             target_labels=target_labels,
                             count=num_training * SC09Dataset.VALIDATION_PERCENTAGE // training_percentage,
                             target_root=target_root,
                             num_components=num_components)

        print('building testing set...')
        self._create_dataset(partition='testing',
                             target_labels=target_labels,
                             count=num_training * SC09Dataset.TESTING_PERCENTAGE // training_percentage,
                             target_root=target_root,
                             num_components=num_components)

    def _create_dataset(self, partition, target_labels, count, target_root, num_components):
        data_dir = os.path.join(target_root, partition)
        os.makedirs(data_dir, exist_ok=True)

        i = 0
        for labels in cycle(target_labels):
            mix_wave = torch.zeros(SC09Dataset.SAMPLE_RATE)
            mix_labels = -torch.ones(num_components, dtype=torch.long)
            mix_sources = []
            for j, label in enumerate(labels):
                if label is None:
                    mix_labels[j] = -1
                else:
                    src_idx, src_wave, _ = next(self.dataloaders[partition][label])
                    rms = src_wave.pow(2).mean().sqrt()
                    src_wave = (src_wave / rms) * self.TRAINING_RMS_MEAN
                    src_wave = src_wave.view(-1) / num_components
                    mix_wave += src_wave
                    mix_labels[j] = label

                    src_magnitude, src_phase = SC09MixtureDataset.spectrogram(src_wave.numpy())
                    mix_sources.append({
                        'wave': src_wave,
                        'magnitude': src_magnitude,
                        # 'phase': src_phase,
                        'label': label,
                    })

            mix_magnitude, mix_phase = SC09MixtureDataset.spectrogram(mix_wave.numpy())

            mixture = {
                'wave': mix_wave,
                'magnitude': mix_magnitude,
                'phase': mix_phase,
                'labels': mix_labels,
                'sources': mix_sources,
            }
            torch.save(mixture, os.path.join(data_dir, '{:07d}.pth'.format(i)))

            if i % 1000 == 0:
                print(i)
            i += 1
            if i >= count:
                break

    @staticmethod
    def _create_dataloaders(sc_root):
        dataloaders = {}
        for partition in SC09Dataset.PARTITIONS:
            by_label = []
            for label in SC09Dataset.LABELS:
                dataset = SC09Dataset(root=sc_root, partition=partition, labels=[label])
                dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
                by_label.append(SC09Mixer._infinite_dataloader(dataloader, label))
            dataloaders[partition] = by_label
        return dataloaders

    @staticmethod
    def _infinite_dataloader(dataloader, label):
        while True:
            i = 0
            for sample in dataloader:
                i += 1
                yield sample
            print('%d samples for label %s exhausted, repeat!' % (i, label))

    @staticmethod
    def _generate_target_labels(classes, num_sources):
        return list(combinations(classes, num_sources))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sc-root', type=str, required=True)
    parser.add_argument('--target-root', type=str, required=True)
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--num-components', type=int, required=True)
    parser.add_argument('--num-training', type=int, required=True)
    args = parser.parse_args()

    mixer = SC09Mixer(args.sc_root)
    mixer.run(target_root=args.target_root,
              classes=list(range(args.num_classes)),
              num_components=args.num_components,
              num_training=args.num_training)


if __name__ == '__main__':
    main()
