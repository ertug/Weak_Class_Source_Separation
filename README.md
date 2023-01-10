# Audio Source Separation Using Variational Autoencoders and Weak Class Supervision
This repository contains supplementary material for [the paper](https://arxiv.org/abs/1810.13104).

## Demo
You can listen to some [**examples of our results**](https://ertug.github.io/Weak_Class_Source_Separation/).

## Source code
- **src/mix.py:** Generates the SC09 mixture dataset.
- **src/train.py:** Training script.
- **src/test.py:** Testing script. It computes the BSS_EVAL metrics.
- **src/experiment.py:** Automates the mix-train-test scripts.
- **src/demo.py:** Generates the demo page.

## Dataset
Speech Commands Dataset v0.02: [**Download**](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)

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
