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
samplesmerged=np.zeros((10,48000*5))
for i in range(16):
    
    samplesmerged[0]+=np.roll(samplesdelayed[i],-int(sampledelay[i][0]))# front
    samplesmerged[1]+=np.roll(samplesdelayed[i],int(sampledelay[i][0]))#back
    samplesmerged[2]+=np.roll(samplesdelayed[i],-int(sampledelay[i][1]))# Left (maybe)
    samplesmerged[3]+=np.roll(samplesdelayed[i],int(sampledelay[i][1]))#Right(maybe)
samplesmerged/=4 
samplesmerged[4]+=np.roll(samplesdelayed[1],int(-5))+ samplesdelayed[6]+   np.roll(samplesdelayed[3],int(-15)) +samplesdelayed[7]
samplesmerged[5]+=np.roll(samplesdelayed[11],int(-5))+ samplesdelayed[9]+   np.roll(samplesdelayed[13],int(-15)) +samplesdelayed[8]
samplesmerged[6]+=np.roll(samplesdelayed[5],int(-30))+ samplesdelayed[10]+   np.roll(samplesdelayed[4],int(-25)) +samplesdelayed[9]
samplesmerged[7]+=np.roll(samplesdelayed[15],int(-30))+ samplesdelayed[0]+   np.roll(samplesdelayed[14],int(-25)) +samplesdelayed[6]
samplesmerged[8]+=np.roll(samplesdelayed[2],int(-26))+ samplesdelayed[10]+   np.roll(samplesdelayed[4],int(-26)) +samplesdelayed[11]
samplesmerged[9]+=np.roll(samplesdelayed[14],int(-26))+ samplesdelayed[2]+   np.roll(samplesdelayed[12],int(-26)) +samplesdelayed[0]
# Now the speical ones  
# 53 degres: 1 6 3 7  dist 15 5
# other 53 : 9 11 8 13 dist 15 5
# mid degrere 5 10   4 9 dist 25.2 30
# other mid : 15 0 14 6 dist 25.2 30
# edge degrre: 2 10 4 11 dist 26 26
# other edge: 14 2 12 0  dist 26 26
samplesmerged/=4

write("ExtraMics16/AudioTests/d1p.wav", 48000,samplesmerged.T)

# Next steps
# All 8(i think) directions right 
# actual testing rather than just eyeballing