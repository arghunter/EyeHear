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
############### Import Modules ###############
from IOStream import IOStream
from SignalGen import SignalGen
from Preprocessor import Preprocessor 
from AudioWriter import AudioWriter
from beamformerMVDR import Beamformer
############### Constants ###############
#SAMPLES_PER_FRAME = 10 #Number of mic reads concatenated within a single window
SAMPLES_PER_FRAME = 100
nfft = 1024#256#1024 #NFFT value for spectrogram
overlap = 512#512 #overlap value for spectrogram
rate = 48000 #sampling rate
sd.default.device=18


im=0
im2=0
ims = []
import soundfile as sf
io=IOStream(20000,20000)
spacing=np.array([[-0.08,0.042],[-0.08,0.014],[-0.08,-0.028],[-0.08,-0.042],[0.08,0.042],[0.08,0.014],[0.08,-0.028],[0.08,-0.042]])
num_microphones=len(spacing)
target_samplerate=48000
sig_gen=SignalGen(num_microphones,spacing,target_samplerate)
speech,samplerate=sf.read(("C:/Users/arg/Documents/Datasets/dev-clean.tar/dev-clean/LibriSpeech/dev-clean/2035/147961/2035-147961-0018.flac"))
interpolator=Preprocessor(mirrored=False,interpolate=int(np.ceil(target_samplerate/16000)))
speech=np.reshape(speech,(-1,1))
speech=interpolator.process(speech)
sig_gen.update_delays(90)
angled_speech=sig_gen.delay_and_gain(speech)
speech1,samplerate=sf.read(("C:/Users/arg/Documents/Datasets/dev-clean.tar/dev-clean/LibriSpeech/dev-clean/652/130737/652-130737-0005.flac"))
interpolator=Preprocessor(mirrored=False,interpolate=int(np.ceil(target_samplerate/16000)))
speech1=np.reshape(speech1,(-1,1))
speech1=interpolator.process(speech1)
sig_gen.update_delays(30)
angled_speech=angled_speech[0:min(len(speech),len(speech1))]+0.6*sig_gen.delay_and_gain(speech1)[0:min(len(speech),len(speech1))]
speech2,samplerate=sf.read(("C:/Users/arg/Documents/Datasets/dev-clean.tar/dev-clean/LibriSpeech/dev-clean/6319/64726/6319-64726-0016.flac"))
interpolator=Preprocessor(mirrored=False,interpolate=int(np.ceil(target_samplerate/16000)))
speech2=np.reshape(speech2,(-1,1))
speech2=interpolator.process(speech2)
sig_gen.update_delays(240)
angled_speech=angled_speech[0:min(len(speech),min(len(speech1),len(speech2)))]+0.5*sig_gen.delay_and_gain(speech2)[0:min(len(speech),min(len(speech1),len(speech2)))]
angled_speech/=np.max(angled_speech)
# angled_speech+=0.05*np.random.randn(*angled_speech.shape)
aw=AudioWriter()
beam=Beamformer(spacing=spacing)
beam.set_doa(90)
# aw1.write("./beamformingarray/AudioTests/Demos/2noise.wav",48000)

io.arrToStream(angled_speech,48000)
# stream=io
# while(not io.complete()):
#     sample=io.getNextSample()
#     # print(sample)
#     aw.add_sample(beam.beamform(sample),480)

streamMVDR=io
# streamMVDR.arrToStream(aw.data,48000)
stream=IOStream()
stream.arrToStream(angled_speech,48000)
############### Functions ###############
"""
get_sample:
gets the audio data from the microphone
inputs: audio stream and PyAudio object
outputs: int16 array
"""
def get_sample():
    data = stream.getNextSample().T[0] *32767 
    # print(data.shape)
    return data
def get_sampleMVDR():
    data = streamMVDR.getNextSample().T[0] *32767 
    # print(data.shape)
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
    ims[0]=im
    data = get_sampleMVDR()
    arr2D,freqs,bins = get_specgram(data,rate)
    im_data = im2.get_array()
    print((stream.q.qsize()))
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

def update_figMVDR(n):
    data = get_sampleMVDR()
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
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
# ax2.plot([1, 2, 3], [6, 5, 4])

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


data2 = get_sampleMVDR()
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


