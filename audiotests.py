from signals import Signal, Sine, Square, Sawtooth, Chirp
import numpy as np
fund = 1500

sig = Square(440)
# sig = 1 / (2*np.pi) * Sine(fund, length=2) + \
#   1 / (4*np.pi) * Sine(2*fund, length=2) + \
#   1 / (6*np.pi) * Sine(3*fund, length=2) + \
#     1 / (8*np.pi) * Sine(4*fund, length=2)
sig.to_wav("sine.wav")
sig.plot_spectrogram()