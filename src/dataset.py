import os

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
import hashlib
import re


class SC09Dataset(Dataset):
    PARTITIONS = ('training', 'validation', 'testing')
    LABELS = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
    SAMPLE_RATE = 8000
    VALIDATION_PERCENTAGE = 10
    TESTING_PERCENTAGE = 10
    MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M

    def __init__(self, root, partition, labels=None):
        if partition not in self.PARTITIONS:
            raise Exception('invalid partition')

        self.metadata = []

        for label in (self.LABELS if labels is None else labels):
            base = os.path.join(root, label)
            filenames = sorted(os.listdir(base))
            for filename in filenames:
                if self._assign_partition(filename) == partition:
                    path = os.path.join(base, filename)
                    self.metadata.append((path, label))

        self.cache = [None for _ in range(len(self.metadata))]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.cache[idx]
        if item is not None:
            return item

        path, label = self.metadata[idx]
        wave = torch.zeros(self.SAMPLE_RATE)
        x, _ = librosa.core.load(path, sr=self.SAMPLE_RATE)

        centered_start = (self.SAMPLE_RATE-x.shape[0])//2
        centered_end = centered_start + x.shape[0]
        wave[centered_start:centered_end] = torch.from_numpy(x)

        item = idx, wave, label
        self.cache[idx] = item
        return item

    def _assign_partition(self, filename):
        """ copied from dataset README """
        base_name = os.path.basename(filename)
        hash_name = re.sub(r'_nohash_.*$', '', base_name)
        hash_name_hashed = hashlib.sha1(hash_name.encode('ascii')).hexdigest()
        percentage_hash = ((int(hash_name_hashed, 16) %
                            (self.MAX_NUM_WAVS_PER_CLASS + 1)) *
                           (100.0 / self.MAX_NUM_WAVS_PER_CLASS))
        if percentage_hash < self.VALIDATION_PERCENTAGE:
            result = 'validation'
        elif percentage_hash < (self.TESTING_PERCENTAGE + self.VALIDATION_PERCENTAGE):
            result = 'testing'
        else:
            result = 'training'
        return result


class SC09MixtureDataset(Dataset):
    PARTITIONS = SC09Dataset.PARTITIONS
    LABELS = SC09Dataset.LABELS
    SAMPLE_RATE = SC09Dataset.SAMPLE_RATE
    NUM_CLASSES = len(SC09Dataset.LABELS)
    FRAME_SIZE = 512
    HOP_SIZE = 256
    CONTEXT_FRAMES = 32
    SPECTROGRAM_BINS = FRAME_SIZE//2 + 1

    def __init__(self, root, partition):
        if partition not in self.PARTITIONS:
            raise Exception('invalid partition')

        self.root = root
        self.partition = partition
        self.data_dir = os.path.join(root, partition)
        self.count = len(os.listdir(self.data_dir))
        self.cache = [None for _ in range(self.count)]

    def _load_mixture(self, idx):
        return torch.load(os.path.join(self.data_dir, '{:07d}.pth'.format(idx)))

    def _get_single_item(self, idx):
        item = self.cache[idx]
        if item is not None:
            return item

        mixture = self._load_mixture(idx)

        mixture['idx'] = idx

        self.cache[idx] = mixture
        return mixture

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self._get_single_item(i) for i in idx]
        else:
            return self._get_single_item(idx)

    @staticmethod
    def spectrogram(wave):
        stft_matrix = librosa.core.stft(y=wave,
                                        n_fft=SC09MixtureDataset.FRAME_SIZE,
                                        hop_length=SC09MixtureDataset.HOP_SIZE)
        magnitude = torch.FloatTensor(np.abs(stft_matrix))
        phase = torch.FloatTensor(np.angle(stft_matrix))
        return magnitude.t(), phase.t()

    @staticmethod
    def magnitude_to_wave(magnitude, phase):
        stft_matrix = magnitude*np.exp(1j*phase.numpy())
        return librosa.core.istft(stft_matrix.T,
                                  hop_length=SC09MixtureDataset.HOP_SIZE,
                                  length=SC09MixtureDataset.SAMPLE_RATE)

    @staticmethod
    def sources_to_waves(sources):
        return torch.stack([source['wave'] for source in sources], dim=1)

    @staticmethod
    def sources_to_magnitudes(sources):
        return torch.stack([source['magnitude'] for source in sources], dim=1)

    @staticmethod
    def magnitudes_to_waves(pred_magnitude, mixture_phase):
        pred_waves = []
        for magnitude_single, mixture_phase_single in zip(pred_magnitude, mixture_phase):
            pred_waves_single = []
            for magnitude_src in magnitude_single:
                pred_waves_single.append(SC09MixtureDataset.magnitude_to_wave(magnitude_src, mixture_phase_single))
            pred_waves.append(np.stack(pred_waves_single))
        return np.stack(pred_waves)
