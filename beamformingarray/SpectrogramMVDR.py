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
from MVDRAsync import MVDRasync
############### Import Modules ###############
from IOStream import IOStream

############### Constants ###############
#SAMPLES_PER_FRAME = 10 #Number of mic reads concatenated within a single window
SAMPLES_PER_FRAME = 60
nfft = 1024#256#1024 #NFFT value for spectrogram
overlap = 512#512 #overlap value for spectrogram
rate = 16000 #sampling rate
sd.default.device=18
stream = IOStream(20000,10000)
stream.streamAudio(rate,8)
stream.getNextSample()
stream.getNextSample()
stream.getNextSample()
stream.getNextSample()
mvdr=MVDRasync(stream)
im=0
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
    data = mvdr.q.get() *32767 
    print(data)
    return data
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
    data = get_sample()
    arr2D,freqs,bins = get_specgram(data,rate)
    im_data = im.get_array()
    print((stream.q.qsize()))
    if n < SAMPLES_PER_FRAME:
        im_data = np.hstack((im_data,arr2D))
        im.set_array(im_data)
    else:
        keep_block = arr2D.shape[1]*(SAMPLES_PER_FRAME - 1)
        im_data = np.delete(im_data,np.s_[:-keep_block],1)
        im_data = np.hstack((im_data,arr2D))
        im.set_array(im_data)
    return im,


############### Initialize Plot ###############
print("Here")
fig = plt.figure()
print("Here")

# """
# Launch the stream and the original spectrogram
# """



data = get_sample()
arr2D,freqs,bins = get_specgram(data,rate)
print("Heree")
"""
Setup the plot paramters
"""
extent = (bins[0],bins[-1]*SAMPLES_PER_FRAME,freqs[-1],freqs[0])
im = plt.imshow(arr2D,aspect='auto',extent = extent,interpolation="none",
                cmap = 'jet',norm = LogNorm(vmin=.01,vmax=1))
print("Hereee")

plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Real Time Spectogram')
plt.gca().invert_yaxis()
##plt.colorbar() #enable if you want to display a color bar

############### Animate ###############
anim = animation.FuncAnimation(fig,update_fig,blit = False,
                            interval=20)

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


