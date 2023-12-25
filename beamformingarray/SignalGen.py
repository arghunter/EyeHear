import numpy as np
from scipy.io.wavfile import write, read
from scipy import signal, interpolate
from scipy.fft import fft, fftfreq
from scipy import interpolate
import matplotlib.pyplot as plt
from pydub import AudioSegment, playback
AUDIO_RATE = 48000

"""
Signal class responsible for all signal processing
"""
class Signal(object):

    def __init__(self, ts, ys, rate=AUDIO_RATE):
        self.rate = rate
        self.ts = ts.astype(np.float32)
        self.ys = ys.astype(np.float32)

    # add two signals of the same size
    def __add__(self, other):
        if self.ys.shape[0] != other.ys.shape[0]:
            raise ValueError(
                f"Dimension Mismatch: {self.ys.shape[0]} != {other.ys.shape[0]}"
            )
        return Signal(self.ts, self.ys + other.ys)

    # subtract two signals of the same size
    def __sub__(self, other):
        if self.ys.shape[0] != other.ys.shape[0]:
            raise ValueError(
                f"Dimension Mismatch: {self.ys.shape[0]} != {other.ys.shape[0]}"
            )
        return Signal(self.ts, self.ys - other.ys)

    # multiply by constant or by eleentwise signal
    def __mul__(self, other):
        # check if other is a number
        if isinstance(other, (int, float)):
            return Signal(self.ts, other * self.ys)
        elif isinstance(other, np.ndarray):
            if self.ys.shape[0] != other.ys.shape[0]:
                raise ValueError(
                    f"Dimension Mismatch: {self.ys.shape[0]} != {other.ys.shape[0]}"
                )
            return Signal(self.ts, np.multiply(self.ys, other))
        return None
    
    def __rmul__(self, other):
        return self.__mul__(other)

    # divide by factor or by elementwise signal
    def __truediv__(self, other):
        # check if other is a number
        if isinstance(other, (int, float)):
            return Signal(self.ts, self.ys / other)
        elif isinstance(other, np.ndarray):
            if self.ys.shape[0] != other.ys.shape[0]:
                raise ValueError(
                    f"Dimension Mismatch: {self.ys.shape[0]} != {other.ys.shape[0]}"
                )
            return Signal(self.ts, np.divide(self.ys, other))
        return None

    # repeat signal n times
    # n - int number of repeats
    def repeat(self, n):
        length = self.ts.shape[0]
        ts = np.linspace(0, n * length, n * length * self.rate, dtype=np.float32)
        ys = np.repeat(self.ys, n)
        return Signal(self.ts, ys)

    # play sound using pydub
    def play(self):
        aseg = AudioSegment(
            self.ys.tobytes(),
            frame_rate=self.rate,
            sample_width=self.ys.dtype.itemsize,
            channels=1
        )
        playback.play(aseg)

    # save as wave file
    def to_wav(self, filename):
        write(filename, self.rate, self.ys)

    # read from wave file
    # filename - string name of file
    @staticmethod
    def from_wav(filename):
        sr, data = read(filename)
        length = data.shape[0] / sr
        ys = data[:, 0] / np.max(np.abs(data[:, 0]), axis=0) # left channel
        ts = np.linspace(0., length, data.shape[0])
        return Signal(ts, ys, rate=sr)

    # calculates the fft of the signal
    def fft(self):
        n = self.ys.shape[0]
        t = 1. / self.rate
        # calculate fft amplitudes
        yf = fft(self.ys, n, )
        # calculate fft frequencies
        xf = fftfreq(n, t)[:n//2]
        return xf, yf

    # plot frequencies in range rng
    # rng - tuple range of frequencies
    def plot_fft(self, rng=(0, 2000)):
        xf, yf = self.fft()
        n = self.ys.shape[0]
        plt.plot(xf, 2.0/n * np.abs(yf[:n//2]))
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        plt.title("FFT")
        plt.xlim(*rng)
        plt.grid()
        plt.show()


    # plot with num_samples, discrete shows only discrete signal
    # num_samples - int number of samples to plot
    # discrete - bool whether to plot on discrete scale
    def plot(self, num_samples=100, discrete=False):
        if discrete:
            plt.scatter(self.ts[:num_samples], self.ys[:num_samples])
        else:
            plt.plot(self.ts[:num_samples], self.ys[:num_samples])
        plt.show()
    
    # plot spectrogram image
    def plot_spectrogram(self):
        f, t, Sxx = signal.spectrogram(self.ys, self.rate)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
        
    # get size of signal in number of samples
    def size(self):
        return self.ys.shape[0]
    
    # get length of sample in seconds
    def length(self):
        return self.ys.shape[0] / self.rate
    
    # get resampled signal from start to end
    def get(self, start, end):
        size = abs(end - start)
        ts = np.linspace(0, size, size * self.rate, dtype=np.float32)
        return Signal(ts, self.ys[start:end+1])

    # filter below or above cutoff
    # cutoff - int cutoff frequency
    # ftype - type of filter
    # order of filter, higher means faster cutoff
    def filter(self, cutoff, ftype="lowpass", order=5):
        sos = signal.butter(order, cutoff, fs=self.rate, btype=ftype, output="sos")
        fy = signal.sosfilt(sos, self.ys)
        return Signal(self.ts, fy)

class Sine(Signal):
    def __init__(self, freq, amp=1., length=1):
        self.ts = np.linspace(0, length, length * AUDIO_RATE, dtype=np.float32)
        self.ys = amp * np.sin(2 * np.pi * freq * self.ts)
        super().__init__(self.ts, self.ys)


class Square(Signal):
    def __init__(self, freq, amp=1., length=1):
        self.ts = np.linspace(0, length, length * AUDIO_RATE, dtype=np.float32)
        self.ys = amp * signal.square(2 * np.pi * freq * self.ts)
        super().__init__(self.ts, self.ys)

class Sawtooth(Signal):
    def __init__(self, freq, amp=1., length=1):
        self.ts = np.linspace(0, length, length * AUDIO_RATE, dtype=np.float32)
        self.ys = amp * signal.sawtooth(2 * np.pi * freq * self.ts)
        super().__init__(self.ts, self.ys)
    

class Chirp(Signal):
    def __init__(self, f0, t1, f1, amp=1., length=1):
        self.ts = np.linspace(0, length, length * AUDIO_RATE, dtype=np.float32)
        self.ys = amp * signal.chirp(self.ts, f0, t1, f1)
        super().__init__(self.ts, self.ys)