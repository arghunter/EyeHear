import numpy as np
from Signal import *
from scipy.io.wavfile import write
from SignalGen import SignalGen

sampledelay=np.array([[0,0],[0,4],[0,10],[0,12],[0,14],[0,18],[3,0],[9,0],[15,0],[21,0],[24,0],[24,4],[24,10],[24,12],[24,14],[24,18]]) 
sig=Sine(1500,0.5,48000)
wave = sig.generate_wave(0.5)
write("ExtraMics16/AudioTests/d1b1.wav", 48000,wave)
sig2=Sine(1000,0.5) #Aliasing threshold is ~1600 on the longest diagonal and the worst case scenario - bunch of testing and math
wave2=sig2.generate_wave(0.5)
wave2=np.roll(wave2,24000)
write("ExtraMics16/AudioTests/d1b2.wav", 48000,wave2)
gen=SignalGen(16,sampledelay*343/48000/1000)
print(gen.spacing)
gen.update_delays(67)
samplesdelayed=gen.delay_and_gain(wave)
gen.update_delays(0)
samplesdelayed+=gen.delay_and_gain(wave2)
# samplesdelayed=np.zeros((16,48000*5))
# for i in range(16):
    
#     samplesdelayed[i]=np.roll(wave,int(sampledelay[i][0]))
noise = np.random.normal(0,1,samplesdelayed.shape)*0.1
# samplesdelayed+=noise
samplesdelayed=samplesdelayed.T

# print(samplesdelayed.dtype)
write("ExtraMics16/AudioTests/d1b.wav", 48000,samplesdelayed.T)
samplesmerged=np.zeros((16,int(48000*0.5)))
for i in range(16):
    
    samplesmerged[0]+=np.roll(samplesdelayed[i],-int(sampledelay[i][0]))# front
    samplesmerged[1]+=np.roll(samplesdelayed[i],int(sampledelay[i][0]))#back
    samplesmerged[2]+=np.roll(samplesdelayed[i],-int(sampledelay[i][1]))# Left (maybe)
    samplesmerged[3]+=np.roll(samplesdelayed[i],int(sampledelay[i][1]))#Right(maybe)
samplesmerged/=4 
samplesmerged[4]+=np.roll(samplesdelayed[1],int(-5))+ samplesdelayed[6]+   np.roll(samplesdelayed[3],int(-15)) +samplesdelayed[7]
samplesmerged[5]+=np.roll(samplesdelayed[1],int(5))+ samplesdelayed[6]+   np.roll(samplesdelayed[3],int(15)) +samplesdelayed[7]
samplesmerged[6]+=np.roll(samplesdelayed[11],int(-5))+ samplesdelayed[9]+   np.roll(samplesdelayed[13],int(-15)) +samplesdelayed[8]
samplesmerged[7]+=np.roll(samplesdelayed[11],int(5))+ samplesdelayed[9]+   np.roll(samplesdelayed[13],int(15)) +samplesdelayed[8]
samplesmerged[8]+=np.roll(samplesdelayed[5],int(-30))+ samplesdelayed[10]+   np.roll(samplesdelayed[4],int(-25)) +samplesdelayed[9]
samplesmerged[9]+=np.roll(samplesdelayed[5],int(30))+ samplesdelayed[10]+   np.roll(samplesdelayed[4],int(25)) +samplesdelayed[9]
samplesmerged[10]+=np.roll(samplesdelayed[15],int(-30))+ samplesdelayed[0]+   np.roll(samplesdelayed[14],int(-25)) +samplesdelayed[6]
samplesmerged[11]+=np.roll(samplesdelayed[15],int(30))+ samplesdelayed[0]+   np.roll(samplesdelayed[14],int(25)) +samplesdelayed[6]
samplesmerged[12]+=np.roll(samplesdelayed[2],int(-26))+ samplesdelayed[10]+   np.roll(samplesdelayed[4],int(-26)) +samplesdelayed[11]
samplesmerged[13]+=np.roll(samplesdelayed[2],int(26))+ samplesdelayed[10]+   np.roll(samplesdelayed[4],int(26)) +samplesdelayed[11]
samplesmerged[14]+=np.roll(samplesdelayed[14],int(-26))+ samplesdelayed[2]+   np.roll(samplesdelayed[12],int(-26)) +samplesdelayed[0]
samplesmerged[15]+=np.roll(samplesdelayed[14],int(26))+ samplesdelayed[2]+   np.roll(samplesdelayed[12],int(26)) +samplesdelayed[0]

# Now the speical ones  
# 53 degres: 1 6 3 7  dist 15 5
# other 53 : 9 11 8 13 dist 15 5
# mid degrere 5 10   4 9 dist 25.2 30
# other mid : 15 0 14 6 dist 25.2 30
# edge degrre: 2 10 4 11 dist 26 26
# other edge: 14 2 12 0  dist 26 26
samplesmerged/=4

write("ExtraMics16/AudioTests/d1p.wav", 48000,samplesmerged.T)

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

# Number of samplepoints
N =int( 48000*0.5)
# sample spacing
T = 1.0 / 48000.0
x = np.linspace(0.0, N*T, N)
y = samplesmerged[15]
yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

fig, ax = plt.subplots()
ax.plot(xf[0:4000], (2.0/N * np.abs(yf[:N//2]))[0:4000])
plt.show()
# Next steps
# Next goal: multiple audio sources and light testing
# 