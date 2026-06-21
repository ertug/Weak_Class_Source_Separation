# Audio Source Separation Using Variational Autoencoders and Weak Class Supervision
This repository contains supplementary material for the paper, available on [arXiv](https://arxiv.org/abs/1810.13104) and [IEEE Xplore](https://ieeexplore.ieee.org/document/8769885).

## Demo
You can listen to some [**examples of our results**](https://ertug.github.io/Weak_Class_Source_Separation/).

## Source code
- **src/mix.py:** Generates the SC09 mixture dataset.
- **src/train.py:** Training script.
- **src/test.py:** Testing script. It computes the BSS_EVAL metrics.
- **src/experiment.py:** Automates the mix-train-test scripts.
- **src/demo.py:** Generates the demo page.
- **notebooks/ResultAnalysis.ipynb:** Analysis of the experiment results.

## Requirements
This project requires Python 3 with the packages pinned in `src/requirements.txt` (PyTorch, SciPy, librosa, mir_eval, matplotlib, Jinja2):

```bash
pip install -r src/requirements.txt
```

## Dataset
SC09 is the subset of spoken digits ("zero"–"nine") from the Speech Commands Dataset v0.02:
<br>
[**Download**](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)

### Attribution
[Speech Commands Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands) (Google; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)) — Pete Warden, *["Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition"](https://arxiv.org/abs/1804.03209)* (2018). The audio samples on the [demo page](https://ertug.github.io/Weak_Class_Source_Separation/) are derived from this dataset and have been modified (mixed, separated, and processed) for this work.

## Usage
The mix → train → test pipeline is driven by `experiment.py` (also see `src/run.sh`):

```bash
cd src
python experiment.py --sc-root ~/datasets/speech_commands --experiments-root ~/experiments --run compare_ae_vae_source_label
```

## Citation (BibTeX)
If you find this repository useful, please cite our work:

```BibTeX
@article{karamatli2019audio,
  title={Audio source separation using variational autoencoders and weak class supervision},
  author={Karamatl{\i}, Ertu{\u{g}} and Cemgil, Ali Taylan and K{\i}rb{\i}z, Serap},
  journal={IEEE Signal Processing Letters},
  volume={26},
  number={9},
  pages={1349--1353},
  year={2019},
  doi={10.1109/LSP.2019.2929440}
}
```
