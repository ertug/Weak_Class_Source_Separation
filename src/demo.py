import os
import argparse
import subprocess
import random

from torch.utils.data import DataLoader
from jinja2 import Environment, PackageLoader, select_autoescape
import librosa

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import SC09MixtureDataset
from models import NVAE
from utils import bss_eval


DEMO_PATH = 'demo'


def generate_image(target, examples_path, name, data):
    plt.figure(figsize=(3, 1.6))
    plt.pcolormesh(data.T)
    plt.axis('off')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    path = os.path.join(examples_path, name + '.jpg')
    plt.savefig(os.path.join(target, path), bbox_inches='tight', pad_inches=0)
    plt.close()
    return path


def generate_audio(target, examples_path, name, data):
    wav_path = os.path.join(target, examples_path, name + '.wav')
    librosa.output.write_wav(path=wav_path,
                             y=data,
                             sr=SC09MixtureDataset.SAMPLE_RATE)

    path = os.path.join(examples_path, name + '.mp3')
    mp3_path = os.path.join(target, path)
    subprocess.check_call(['lame', '-q0', '-V0', wav_path, mp3_path],
                          stdin=subprocess.DEVNULL,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
    os.remove(wav_path)
    return path


def generate_pair(target, examples_path, name, image_data, audio_data):
    return {
        'image': generate_image(target=target,
                                examples_path=examples_path,
                                name=name,
                                data=image_data),
        'audio': generate_audio(target=target,
                                examples_path=examples_path,
                                name=name,
                                data=audio_data)
    }


def generate_examples(target, sc09_mixture_root, checkpoints, step, num_examples):
    dataset = SC09MixtureDataset(root=sc09_mixture_root, partition='testing')
    datasubset = random.sample(list(dataset), num_examples)

    dataloader = DataLoader(datasubset, batch_size=num_examples, shuffle=False)
    samples = next(iter(dataloader))

    mixture_magnitude = samples['magnitude']
    mixture_wave = samples['wave']
    mixture_sources = samples['sources']
    mixture_labels = samples['labels']
    mixture_phase = samples['phase']
    num_sources = len(mixture_sources)

    nvae = NVAE.load(path=os.path.join(checkpoints, 'train', '{:06d}.pth'.format(step)),
                     device='cpu')

    nvae.forward(x_mix=mixture_magnitude,
                 x_src=SC09MixtureDataset.sources_to_magnitudes(mixture_sources),
                 y_sparse=samples['labels'],
                 train=False)

    pred_x_src_masked, pred_x_src_mask = nvae.batch_p_x_src_masked()
    pred_x_src_masked = pred_x_src_masked.detach()
    pred_x_src_mask = pred_x_src_mask.detach()
    pred_x_src_masked_wave = SC09MixtureDataset.magnitudes_to_waves(pred_x_src_masked.numpy(), mixture_phase)

    ref_waves = SC09MixtureDataset.sources_to_waves(mixture_sources).numpy()
    bss_eval_results = bss_eval(ref_waves, pred_x_src_masked_wave)

    examples = []

    for i in range(num_examples):
        print(mixture_labels[i].numpy().tolist())

        example = {}
        examples_path = os.path.join('examples', '{:03d}'.format(i))
        os.makedirs(os.path.join(target, examples_path), exist_ok=True)

        example['labels'] = mixture_labels[i].numpy().tolist()
        example['bss_eval'] = {
            'sdr': bss_eval_results[i][0].tolist(),
            'sir': bss_eval_results[i][1].tolist(),
            'sar': bss_eval_results[i][2].tolist(),
        }

        image_data = librosa.core.amplitude_to_db(mixture_magnitude[i].detach().numpy())
        audio_data = mixture_wave[i].numpy()
        example['mixture'] = generate_pair(target=target,
                                           examples_path=examples_path,
                                           name='mixture',
                                           image_data=image_data,
                                           audio_data=audio_data)

        example['masks'] = []
        for j in range(num_sources):
            data = pred_x_src_mask[i, j].detach().numpy()
            example['masks'].append(generate_image(target=target,
                                                   examples_path=examples_path,
                                                   name='mask-%d' % j,
                                                   data=data))

        example['sources_true'] = []
        for j in range(num_sources):
            image_data = librosa.core.amplitude_to_db(mixture_sources[j]['magnitude'][i].numpy())
            audio_data = mixture_sources[j]['wave'][i].numpy()
            example['sources_true'].append(generate_pair(target=target,
                                                         examples_path=examples_path,
                                                         name='source-true-%d' % j,
                                                         image_data=image_data,
                                                         audio_data=audio_data))

        example['sources_pred'] = []
        for j in range(num_sources):
            image_data = librosa.core.amplitude_to_db(pred_x_src_masked[i, j].detach().numpy())
            audio_data = pred_x_src_masked_wave[i, j]
            example['sources_pred'].append(generate_pair(target=target,
                                                         examples_path=examples_path,
                                                         name='source-pred-%d' % j,
                                                         image_data=image_data,
                                                         audio_data=audio_data))

        examples.append(example)

    return examples


def generate(target, sc09_mixture_root, checkpoints, step, num_examples, seed):
    examples = generate_examples(target, sc09_mixture_root, checkpoints, step, num_examples)

    env = Environment(
        loader=PackageLoader('demo', DEMO_PATH),
        autoescape=select_autoescape(['html'])
    )

    template = env.get_template('template.html')
    html = template.render(examples=examples,
                           seed=seed)

    html_path = os.path.join(target, 'index.html')
    with open(html_path, encoding='utf-8', mode='w') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sc09-mixture-root', type=str, required=True)
    parser.add_argument('--checkpoints', type=str, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--num-examples', type=int, default=10)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    os.makedirs(args.target, exist_ok=True)
    if os.listdir(args.target):
        raise Exception('target dir "%s" is not empty!' % args.target)

    random.seed(args.seed)

    generate(target=args.target,
             sc09_mixture_root=args.sc09_mixture_root,
             checkpoints=args.checkpoints,
             step=args.step,
             num_examples=args.num_examples,
             seed=args.seed)


if __name__ == '__main__':
    main()
