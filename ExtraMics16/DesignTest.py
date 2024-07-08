import numpy as np
from Signal import *
from scipy.io.wavfile import write
sampledelay=np.array([[0,0],[0,4],[0,10],[0,12],[0,14],[0,18],[3,0],[9,0],[15,0],[21,0],[24,0],[24,4],[24,10],[24,12],[24,14],[24,18]])
sig=Sine(250,0.8,48000)
wave = sig.generate_wave(5).reshape((48000*5))

samplesdelayed=np.zeros((1,48000*5))
for i in range(1):
    
    samplesdelayed[i]=np.roll(wave,int(sampledelay[i][0])).astype(np.float16)
noise = np.random.normal(0,0.5,samplesdelayed.shape)
# samplesdelayed+=noise
print(samplesdelayed.dtype)
write("ExtraMics16/AudioTests/d1.wav", 48000,samplesdelayed.astype(np.float16))