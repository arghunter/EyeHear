import numpy as np
from Signal import *
from scipy.io.wavfile import write
sampledelay=np.array([[0,0],[0,4],[0,10],[0,12],[0,14],[0,18],[3,0],[9,0],[15,0],[21,0],[24,0],[24,4],[24,10],[24,12],[24,14],[24,18]])
sig=Square(1,0.5,48000)
wave = sig.generate_wave(5).reshape((48000*5))

samplesdelayed=np.zeros((16,48000*5))
for i in range(16):
    
    samplesdelayed[i]=np.roll(wave,int(sampledelay[i][0]))
noise = np.random.normal(0,1,samplesdelayed.shape)*0.2
samplesdelayed+=noise
# print(samplesdelayed.dtype)
write("ExtraMics16/AudioTests/d1b.wav", 48000,samplesdelayed.T)
samplesmerged=np.zeros((8,48000*5))
for i in range(16):
    
    samplesmerged[0]+=np.roll(samplesdelayed[i],-int(sampledelay[i][0]))# front
    samplesmerged[1]+=np.roll(samplesdelayed[i],int(sampledelay[i][0]))#back
    samplesmerged[2]+=np.roll(samplesdelayed[i],-int(sampledelay[i][1]))# Left (maybe)
    samplesmerged[3]+=np.roll(samplesdelayed[i],int(sampledelay[i][1]))#Right(maybe)
    
# Now the speical ones  
# 53 degres: 1 6 3 7

    
samplesmerged/=16
write("ExtraMics16/AudioTests/d1p.wav", 48000,samplesmerged.T)

# Next steps
# All 8(i think) directions right 
# actual testing rather than just eyeballing