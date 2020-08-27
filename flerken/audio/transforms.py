import numpy as np

import torch
from torch import cat, atan2

__all__ = ['LogFrequencyScale', 'STFT', 'rec2polar']


class LogFrequencyScale(torch.nn.Module):
    """
    Rescale linear spectrogram into log-frequency spectrogram. Namely, maps linear values into a log scale
     without modifiying them


    :param warp: True for going from linear to log, false otherwise
    :type warp: bool
    :param ordering: Spectrogram's channel order. By default 'BHWC' (Batch, H,W, channels) according to standard
    pytorch spectrogram ordering --> B,H,W,[real,imag]. Any ordering allowed eg. 'HW', 'HCWB', 'CBWH', 'HBW'
    :type ordering: str
    :param shape: Optional resizing (H_desired, W_desired). (None,None) by default --> without resizing.
    :type shape: tuple
    :param kwargs: Aditional arguments which will be parsed to pytorch's gridsample
    """

    def __init__(self, warp: bool, ordering: str = 'BHWC', shape: tuple = (None, None), adaptative=False, **kwargs):
        super().__init__()
        self.expected_dim = len(ordering)
        self.warp = warp
        self.ordering = ordering.lower()
        self.var = self.get_dims(self.ordering)
        self.instantiated = False
        self.adaptative = adaptative
        self.exp_ordering = ''.join(sorted(self.ordering))  # Expected ordering
        self.kwargs = kwargs
        self.shape = shape

    def needtoinstantiate(self):
        return (not self.instantiated) | self.adaptative

    @staticmethod
    def get_dims(ordering):
        var = {'b': None, 'h': None, 'w': None, 'c': None}
        assert 'h' in ordering
        assert 'w' in ordering
        for key in var:
            var[key] = ordering.find(key)
        return var

    def instantiate(self, sp):
        self.instantiated = True
        dims = sp.shape
        H = dims[self.var['h']] if self.shape[0] is None else self.shape[0]
        W = dims[self.var['w']] if self.shape[1] is None else self.shape[1]
        B = dims[self.var['b']] if self.var['b'] != -1 else 1
        # self.grid = self.get_grid(B, H, W).to(sp.device)
        self.register_buffer('grid', self.get_grid(B, H, W).to(sp.device))
        self.squeeze = []
        if self.var['b'] == -1:
            self.squeeze.append(0)
        if self.var['c'] == -1:
            self.squeeze.append(1)

    def get_grid(self, bs, HO, WO):
        # meshgrid
        x = np.linspace(-1, 1, WO)
        y = np.linspace(-1, 1, HO)
        xv, yv = np.meshgrid(x, y)
        grid = np.zeros((bs, HO, WO, 2))
        grid_x = xv
        if self.warp:
            grid_y = (np.power(21, (yv + 1) / 2) - 11) / 10
        else:
            grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
        grid[:, :, :, 0] = grid_x
        grid[:, :, :, 1] = grid_y
        grid = grid.astype(np.float32)
        return torch.from_numpy(grid)

    def forward(self, sp):
        """
        :param sp: Spectrogram decribed at :class:`.LogFrequencyScale`
        :type sp: torch.Tensor
        :return: Transformed spectrogram
        """
        if self.needtoinstantiate():
            self.instantiate(sp)
        sp = torch.einsum(self.ordering + '->' + self.exp_ordering, sp)
        for dim in self.squeeze:
            sp.unsqueeze_(dim)
        sp = torch.nn.functional.grid_sample(sp, self.grid, **self.kwargs)
        for dim in self.squeeze[::-1]:
            sp.squeeze_(dim)
        sp = torch.einsum(self.exp_ordering + '->' + self.ordering, sp)
        return sp


class STFT(object):
    def __init__(self, window=torch.hamming_window, normalization: bool = False, n_fft: int = 1024,
                 hop_length=None, win_length=None,
                 center=True, pad_mode='reflect', normalized=False, onesided=True):
        r"""Short-time Fourier transform (STFT).

        Ignoring the optional batch dimension, this method computes the following
        expression:

        .. math::
            X[m, \omega] = \sum_{k = 0}^{\text{win\_length-1}}%
                                \text{window}[k]\ \text{input}[m \times \text{hop\_length} + k]\ %
                                \exp\left(- j \frac{2 \pi \cdot \omega k}{\text{win\_length}}\right),

        where :math:`m` is the index of the sliding window, and :math:`\omega` is
        the frequency that :math:`0 \leq \omega < \text{n\_fft}`. When
        :attr:`onesided` is the default value ``True``,

        * :attr:`input` must be either a 1-D time sequence or a 2-D batch of time
          sequences.

        * If :attr:`hop_length` is ``None`` (default), it is treated as equal to
          ``floor(n_fft / 4)``.

        * If :attr:`win_length` is ``None`` (default), it is treated as equal to
          :attr:`n_fft`.

        * :attr:`window` can be a 1-D tensor of size :attr:`win_length`, e.g., from
          :meth:`torch.hann_window`. If :attr:`window` is ``None`` (default), it is
          treated as if having :math:`1` everywhere in the window. If
          :math:`\text{win\_length} < \text{n\_fft}`, :attr:`window` will be padded on
          both sides to length :attr:`n_fft` before being applied.

        * If :attr:`center` is ``True`` (default), :attr:`input` will be padded on
          both sides so that the :math:`t`-th frame is centered at time
          :math:`t \times \text{hop\_length}`. Otherwise, the :math:`t`-th frame
          begins at time  :math:`t \times \text{hop\_length}`.

        * :attr:`pad_mode` determines the padding method used on :attr:`input` when
          :attr:`center` is ``True``. See :meth:`torch.nn.functional.pad` for
          all available options. Default is ``"reflect"``.

        * If :attr:`onesided` is ``True`` (default), only values for :math:`\omega`
          in :math:`\left[0, 1, 2, \dots, \left\lfloor \frac{\text{n\_fft}}{2} \right\rfloor + 1\right]`
          are returned because the real-to-complex Fourier transform satisfies the
          conjugate symmetry, i.e., :math:`X[m, \omega] = X[m, \text{n\_fft} - \omega]^*`.

        * If :attr:`normalized` is ``True`` (default is ``False``), the function
          returns the normalized STFT results, i.e., multiplied by :math:`(\text{frame\_length})^{-0.5}`.

        Returns the real and the imaginary parts together as one tensor of size
        :math:`(* \times N \times T \times 2)`, where :math:`*` is the optional
        batch size of :attr:`input`, :math:`N` is the number of frequencies where
        STFT is applied, :math:`T` is the total number of frames used, and each pair
        in the last dimension represents a complex number as the real part and the
        imaginary part.

        .. warning::
          This function changed signature at version 0.4.1. Calling with the
          previous signature may cause error or return incorrect result.

        Arguments:
            input (Tensor): the input tensor
            n_fft (int): size of Fourier transform
            hop_length (int, optional): the distance between neighboring sliding window
                frames. Default: ``None`` (treated as equal to ``floor(n_fft / 4)``)
            win_length (int, optional): the size of window frame and STFT filter.
                Default: ``None``  (treated as equal to :attr:`n_fft`)
            window (Tensor, optional): the optional window function.
                Default: ``None`` (treated as window of all :math:`1` s)
            center (bool, optional): whether to pad :attr:`input` on both sides so
                that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
                Default: ``True``
            pad_mode (string, optional): controls the padding method used when
                :attr:`center` is ``True``. Default: ``"reflect"``
            normalized (bool, optional): controls whether to return the normalized STFT results
                 Default: ``False``
            normalization (bool, optional): normalize output dividing by window.sum()
                 Default: ``False``
            onesided (bool, optional): controls whether to return half of results to
                avoid redundancy Default: ``True``

        Returns:
            Tensor: A tensor containing the STFT result with shape described above

        """
        self.window = window(n_fft)
        self.norm_coef = self.window.sum().item()
        self.normalization = normalization
        self.n_fft = n_fft

        self.hop_length = hop_length
        self.win_length = win_length

        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided

    def __call__(self, raw_wf):
        if raw_wf.device != self.window.device:
            window = self.window.to(raw_wf.device)
        else:
            window = self.window
        stft = torch.stft(raw_wf, window=window, n_fft=self.n_fft,
                          hop_length=self.hop_length, win_length=self.win_length,
                          center=self.center, pad_mode=self.pad_mode, normalized=self.normalized,
                          onesided=self.onesided)
        if self.normalization:
            return stft / self.norm_coef
        else:
            return stft


def rec2polar(tensor, dim=None):
    if dim == None:
        dim = tensor.dim() - 1
    assert tensor.size(dim) == 2  # Re and Imag
    mag = tensor.norm(dim=dim).unsqueeze(dim)
    re, imag = tensor.chunk(2, dim)
    phase = atan2(imag, re)

    return cat([mag, phase], dim=dim)
