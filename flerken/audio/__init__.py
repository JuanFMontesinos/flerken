import librosa
import librosa.display as disp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from . import transforms

__all__ = ['transforms', 'plot_sp']


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


def plot_sp(self, yaxis):
    plt.figure()
    D = librosa.amplitude_to_db(self.sp, ref=np.max)
    disp.specshow(D, x_axis='time', y_axis=yaxis)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Time-frequency power spectrogram')
    plt.show()


def test():
    filename = librosa.util.example_audio_file()
    y, sr = librosa.load(filename)
    D = np.abs(librosa.stft(y))
