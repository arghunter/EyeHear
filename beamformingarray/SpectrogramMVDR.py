"""
run_specgram.py
Created By Alexander Yared (akyared@gmail.com)

Main Script for the Live Spectrogram project, a real time spectrogram
visualization tool

Dependencies: matplotlib, numpy and the mic_read.py module
"""
############### Import Libraries ###############
from matplotlib.mlab import window_hanning,specgram
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import numpy as np
import sounddevice as sd
from queue import Queue
from MVDRAsync import MVDRasync
############### Import Modules ###############
from IOStream import IOStream
from IOStream import IOStream
from SignalGen import SignalGen
from Preprocessor import Preprocessor 
from AudioWriter import AudioWriter
from beamformerMVDR import Beamformer
############### Constants ###############
#SAMPLES_PER_FRAME = 10 #Number of mic reads concatenated within a single window
SAMPLES_PER_FRAME = 120
q=Queue()
buffer=np.zeros((2,480,8))
buffer_head=np.zeros((1))


nfft = 1024#256#1024 #NFFT value for spectrogram
overlap = 512#512 #overlap value for spectrogram
rate = 8000 #sampling rate
sd.default.device=18
# Get both running in one plot
# stream.streamAudio(48000,8)      

# stream.getNextSample()
# stream.getNextSample()
# stream.getNextSample()
# stream.getNextSample()
def audio_callback(indata, frames, time, status):
        # print(("1"))
    # q.qsize()
    for i in range(frames):
        
        if(buffer_head[0]<buffer.shape[1]):
            buffer[0,int(buffer_head[0])]=indata[i]
            buffer_head[0]+=1
        else:
           q.put(np.concatenate([buffer[1],buffer[0]]))
        #    q.put(buffer[0])
           buffer[1]=buffer[0] 
           buffer_head[0]=0


stream=sd.InputStream(
            device=None, channels=8,
            samplerate=8000, callback=audio_callback)
stream.start()

# mvdr=MVDRasync(stream)
im=0
im2=0
ims = []
q.get()
q.get()
q.get()
q.get()
q.get()
beam=MVDRasync(q)
# Probably in delayt
# for i in range(10):
#     print(stream.getNextSample().shape)
############### Functions ###############
"""
get_sample:
gets the audio data from the microphone
inputs: audio stream and PyAudio object
outputs: int16 array
"""
def get_sample():
    # data = q.get().T[0].T *32767 
    print(q.qsize())
    data=beam.q.get()
    frame=beam.dq.get()
    
    return frame,data
"""
get_specgram:
takes the FFT to create a spectrogram of the given audio signal
input: audio signal, sampling rate
output: 2D Spectrogram Array, Frequency Array, Bin Array
see matplotlib.mlab.specgram documentation for help
"""

"""
update_fig:
updates the image, just adds on samples at the start until the maximum size is
reached, at which point it 'scrolls' horizontally by determining how much of the
data needs to stay, shifting it left, and appending the new data. 
inputs: iteration number
outputs: updated image
"""

def get_specgram(signal,rate):
    arr2D,freqs,bins = specgram(signal,window=window_hanning,
                                Fs = rate,NFFT=nfft,noverlap=overlap)
    return arr2D,freqs,bins
def update_fig(n):
    
        
    frame,data = get_sample()
    arr2D,freqs,bins = get_specgram(32767*frame,rate)
    im_data = im.get_array()
    # print((stream.q.qsize()))
    if n < SAMPLES_PER_FRAME:
        im_data = np.hstack((im_data,arr2D))
        im.set_array(im_data)
    else:
        keep_block = arr2D.shape[1]*(SAMPLES_PER_FRAME - 1)
        im_data = np.delete(im_data,np.s_[:-keep_block],1)
        im_data = np.hstack((im_data,arr2D))
        im.set_array(im_data)
    ims[0]=im
    
    arr2D,freqs,bins = get_specgram(data,rate)
    im_data = im2.get_array()
    # print((stream.q.qsize()))
    if n < SAMPLES_PER_FRAME:
        im_data = np.hstack((im_data,arr2D))
        im.set_array(im_data)
    else:
        keep_block = arr2D.shape[1]*(SAMPLES_PER_FRAME - 1)
        im_data = np.delete(im_data,np.s_[:-keep_block],1)
        im_data = np.hstack((im_data,arr2D))
        im2.set_array(im_data)
    ims[1]=im2
    return ims


############### Initialize Plot ###############
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
# ax2.plot([1, 2, 3], [6, 5, 4])

print("Here")

# """
# Launch the stream and the original spectrogram
# """



data,frame = get_sample()
arr2D,freqs,bins = get_specgram(data,rate)
print("Heree")
"""
Setup the plot paramters
"""
extent = (bins[0],bins[-1]*SAMPLES_PER_FRAME,freqs[-1],freqs[0])
im = ax1.imshow(arr2D,aspect='auto',extent = extent,interpolation="none",
                cmap = 'jet',norm = LogNorm(vmin=.01,vmax=1))
print("Hereee")

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Frequency (Hz)')
ax1.set_title('Real Time Spectogram')
ax1.invert_yaxis()
# plt.gca().invert_yaxis()

##plt.colorbar() #enable if you want to display a color bar

############### Animate ###############


frame,data2 = get_sample()
arr2D2,freqs2,bins2 = get_specgram(data2,rate)
print("Heree")
"""
Setup the plot paramters
"""
extent = (bins2[0],bins2[-1]*SAMPLES_PER_FRAME,freqs2[-1],freqs2[0])
im2 = ax2.imshow(arr2D2,aspect='auto',extent = extent,interpolation="none",
                cmap = 'jet',norm = LogNorm(vmin=.01,vmax=1))
print("Hereee")

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Frequency (Hz)')
ax2.set_title('Real Time Spectogram')
ax2.invert_yaxis()
ims=[im,im2]
##plt.colorbar() #enable if you want to display a color bar

############### Animate ###############
anim2 = animation.FuncAnimation(fig,update_fig,blit = False,
                            interval=20)
# anim = animation.FuncAnimation(fig,update_fig,blit = False,
#                             interval=20)
print("Hereee")
                          
try:
    plt.show()
except:
    print("Plot Closed")

##plt.colorbar() #enable if you want to display a color bar

# ############### Animate ###############
# anim = animation.FuncAnimation(fig,update_fig,blit = False,
#                             interval=20)

          
# try:
#     plt.show()
# except:
#     print("Plot Closed")

# ############### Terminate ###############


# print("Program Terminated")


