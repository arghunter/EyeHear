import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def generate_signal(fs, duration):
    t = np.arange(0, duration, 1/fs)
    signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.random.normal(size=len(t))
    return  signal
# img=0
import matplotlib.pyplot as plt
from matplotlib import animation
import librosa
import sounddevice as sd
WINDOW = 100_000
JUMP = 1000
INTERVAL = 20
FILENAME = 'sound.wav'

sound = generate_signal(48000,5)
rate=48000
fig = plt.figure()
from IOStream import IOStream
sd.default.device=9
io=IOStream(frame_shift=20000)
io.wavToStream("./beamformingarray/AudioTests/10.wav")
# io.streamAudio(48000,1)
lc=[0]
def animate(i):
    global lc
    # print(io.getNextSample())
    # if lc!=0:
    #     chunk = np.concatenate([chunk,io.getNextSample()])
    # else:
    #     chunk=io.getNextSample()
    lc=np.concatenate([lc,io.getNextSample()])
    
    _, _, _, im = plt.specgram(lc, Fs=rate)
   
    return im,


ani = animation.FuncAnimation(fig, animate, interval=INTERVAL, blit=True)

# plt.ion()
plt.show()