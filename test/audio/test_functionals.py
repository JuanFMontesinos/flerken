import unittest

import torch
import numpy as np
import librosa

from flerken import audio
from flerken.audio.transforms import rec2polar


class TestAudioFunctionals(unittest.TestCase):
    def test_torch_binary_mask(self):
        N = 65535
        n = 2
        waveforms = [torch.rand(N) for _ in range(n)]
        spectrograms = [torch.stft(x, 1022, 256, window=torch.hann_window(1022)) for x in waveforms]
        spm_with_phase, spm, sp, gt = audio.torch_binary_max(spectrograms, False)

        spm_with_phase_ = sum(spectrograms)
        sp_ = torch.stack([rec2polar(s) for s in spectrograms])[..., 0]
        spm_ = rec2polar(spm_with_phase)[..., 0]

        gt_ = torch.stack([(sp_[0] > sp_[1]).float(), (sp_[1] > sp_[0]).float()])

        self.assertTrue((spm_with_phase == spm_with_phase_).all())

        self.assertTrue((spm == spm_).all())
        self.assertTrue((sp == sp_).all())
        self.assertTrue((gt == gt_).all())

        with self.assertRaises(ValueError):
            spectrograms[0][0, ...] = float('NaN')
            spm_with_phase, spm, sp, gt = audio.torch_binary_max(spectrograms, True)
        with self.assertRaises(ValueError):
            spectrograms[0][0, ...] = float('inf')
            spm_with_phase, spm, sp, gt = audio.torch_binary_max(spectrograms, True)

    def test_numpy_binary_mask(self):
        N = 65535
        n = 2
        waveforms = [np.random.rand(N) for _ in range(n)]
        spectrograms = [librosa.stft(x, 1022, 256) for x in waveforms]
        spm_with_phase, spm, sp, gt = audio.numpy_binary_max(spectrograms)

        spm_with_phase_ = sum(spectrograms)
        sp_ = np.stack([np.abs(s) for s in spectrograms])
        spm_ = np.abs(spm_with_phase_)

        gt_ = np.stack([(sp_[0] > sp_[1]).astype(np.float32), (sp_[1] > sp_[0]).astype(np.float)])

        self.assertTrue((spm_with_phase == spm_with_phase_).all())
        self.assertTrue((spm == spm_).all())
        self.assertTrue((sp == sp_).all())
        self.assertTrue((gt == gt_).all())

