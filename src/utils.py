import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import mir_eval

EPS = 1e-20


def common_init():
    seed = 123
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = True


def total_params(params):
    return sum(np.prod(param.size()) for param in params)


def soft_mask(x_src, x_mix):
    mask = x_src.pow(2) / (x_src.pow(2).sum(dim=1, keepdim=True) + EPS)
    x_src_masked = (x_mix.unsqueeze(dim=1) + EPS) * mask
    return x_src_masked, mask


def bss_eval(ref_waves, pred_waves):
    results = []
    for ref_waves_single, pred_waves_single in zip(ref_waves, pred_waves):
        pred_sdr, pred_sir, pred_sar, _ = mir_eval.separation.bss_eval_sources(ref_waves_single, pred_waves_single,
                                                                               compute_permutation=False)
        results.append(np.stack([pred_sdr, pred_sir, pred_sar]))
    return np.stack(results)
