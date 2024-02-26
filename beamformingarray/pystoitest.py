import soundfile as sf
from pystoi import stoi

denoised, fs = sf.read('./beamformingarray/AudioTests/10.wav')
clean, fs = sf.read('./beamformingarray/AudioTests/test_input_sig.wav')
denoised=denoised[0:134640]
clean=(clean[0:134640]).T[0].T
# Clean and den should have the same length, and be 1D
d = stoi(clean, denoised, fs, extended=False)
print(d)