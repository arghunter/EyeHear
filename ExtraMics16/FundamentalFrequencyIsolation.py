import numpy as np
from Signal import *
from scipy.io.wavfile import write,read
from SignalGen import SignalGen
from Preprocessor import Preprocessor
import soundfile as sf
from VAD import VAD
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal
speech,samplerate=sf.read(("C:/Users/arg/Documents/Datasets/dev-clean.tar/dev-clean/LibriSpeech/dev-clean/2035/147961/2035-147961-0018.flac"))
chunk_size=int(samplerate/100);
end=chunk_size;
start=0
vad=VAD()
while end<=len(speech):

    if vad.is_speech(speech[start:end]) :
        # Number of samplepoints
        N =int( chunk_size)
        # sample spacing
        T = 1.0 / samplerate
        x = np.linspace(0.0, N*T, N)
        y = speech[start:end]
        yf = scipy.fftpack.fft(y)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        magnitude = 2.0 / N * np.abs(yf[:N // 2])
        peaks, _ = scipy.signal.find_peaks(magnitude)
        first_peak_freq = xf[peaks[0]]

        # Find the multiples of the first peak
        multiples = []
        for multiple in range(2, int(xf[-1] // first_peak_freq) + 1):
            multiple_freq = multiple * first_peak_freq
            closest_idx = np.argmin(np.abs(xf - multiple_freq))
            multiples.append(closest_idx) 
        fig, ax = plt.subplots()
        ax.plot(xf, magnitude)
        ax.plot(xf[peaks], magnitude[peaks], 'ro', label='Peaks')
        ax.plot(xf[multiples], magnitude[multiples], 'go', label='Multiples of First Peak')  # Highlight multiples with green x's
        for i in multiples:
            magnitude[i]=0
        ax.legend()
        plt.show()
    end+=chunk_size
    start+=chunk_size
    print(end/len(speech))