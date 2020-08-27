from typing import List

import librosa
import librosa.display as disp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import torch

from . import transforms

__all__ = ['transforms']


# NUMPY FUNCTIONS

def lin2log_powerfunction(x, b):
    N = len(x)
    sr = max(x)
    a = (sr / b + 1) ** (1 / N)
    assert a > 1
    print(a)
    c = -b

    def inner(x):
        return b * a ** x + c

    return inner


def lin2log(spectrogram, sr=11025, n_fft=1022):
    ax_freq_lin = librosa.fft_frequencies(sr=2 * sr, n_fft=n_fft)
    f = lin2log_powerfunction(ax_freq_lin, 30)
    ax_freq_log = f(np.arange(len(ax_freq_lin)))
    set_interp = interp1d(ax_freq_lin, spectrogram, kind='linear', axis=0)
    result = set_interp(ax_freq_log)
    return result[result.shape[0] // 2:]


def numpy_binary_max(sp: List[np.array]):
    """
    Computes the binary mask of a ordered list of spectrograms.
    :param sp: Spectrogram of type np.complex64
    :type sp: np.ndarray
    :return spm_with_phase:

    """
    spm_with_phase = sum(sp)
    sp = np.stack(sp)
    sp = np.abs(sp)

    spm = np.abs(spm_with_phase)

    indices = sp.argmax(axis=0)
    gt = np.zeros_like(sp)
    for idx, sp_i in enumerate(gt):
        mask = indices == idx
        sp_i[mask] = 1.

    return spm_with_phase, spm, sp, gt


# TORCH FUNCTIONS

def torch_binary_max(sp: List[torch.tensor], debug: bool):
    """
    Computes the binary masks for spectrograms matching the format of torch.stft
    :param sp: Ordered list of torch spectrograms
    :type sp: list
    :param debug: Flag which enables NaN and Inf detection
    :type debug: bool
    :return:
    """
    sp = torch.stack(sp)
    spm_with_phase = sum(sp)
    spm = transforms.rec2polar(spm_with_phase)[..., 0]  # unsqueeze to fit required channel dim in conv2d
    if debug:
        for i in sp:
            if torch.isnan(i).any() or torch.isinf(i).any():
                raise ValueError('NaN or Inf found in raw spectrograms.')
        if torch.isnan(spm).any() or torch.isinf(spm).any():
            raise ValueError('NaN or Inf found in spm after getting mag and phase.')
    sp = torch.stack([transforms.rec2polar(s) for s in sp])[..., 0]
    indices = sp.max(dim=0)[1]
    gt = torch.zeros_like(sp)
    for idx, sp_i in enumerate(gt):
        sp_i.masked_fill_(indices == idx, 1.)

    return spm_with_phase, spm, sp, gt
