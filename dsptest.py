

import sounddevice as sd
from scipy.io.wavfile import write

fs = 48000  # Sample rate
seconds = 4  # Duration of recording
sd.default.device=33
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=8)
sd.wait()  # Wait until recording is finished

write('./beamformingarray/AudioTests/clear1.wav', fs, myrecording)  # Save as WAV file 
