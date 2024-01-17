import numpy as np
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import scipy.signal as signal
from time import time as time1

# io= IOStream(sample_duration=20000)
# io.wavToStream("./beamformingarray/AudioTests/test_input_sig.wav")
file = read("./beamformingarray/AudioTests/test_input_sig_2_1.wav")

pcm=np.array(file[1])/32767
print(np.max(pcm))
FS=file[0]
N=pcm.shape[0]
stft_len=1024
num_channel=pcm.shape[1]
frame_len=960
frame_shift=480 # 50% overlap
exp_avg_param=50
C=343.3
d=0.028
theta=0
win=np.hanning(frame_len)
win=win*frame_shift/sum(win) # why?
win_multi=np.tile(win,(num_channel,1)).T
frame_num=int(np.floor((N-frame_len)/frame_shift+1));
# print(frame_num)
N=31000
output=np.zeros((N,1))
N_f=int(stft_len/2)+1#Num frequencies
global_covar=np.zeros((num_channel,num_channel,N_f),dtype='complex128')
frame_count=1
i=0
mu=0
# while i + frame_len < N:# while i<=0:# 
#     t1=int(time1() * 1000)
#     # print(pcm[j : j + frame_len,:].shape)
#     win_data = np.asmatrix(pcm[i : i + frame_len,:]*win_multi)
#     # print(win_data)
#     spectrum=np.asmatrix(np.fft.rfft(win_data,stft_len,axis=0))
#     # print(spectrum.shape)
#     # print(win_data.T[0].T.shape)
    
#     X = np.fft.fft(win_data.T[0].T,axis=0);

#     Y = np.fft.fft(win_data.T[0].T,axis=0);

#     R = np.multiply(X,np.conj(Y));

#     tphat = np.real(np.fft.ifft(np.divide(R,np.abs(R)),axis=0));
#     tphat=np.reshape(tphat,(-1))
#     locs, _ = signal.find_peaks(tphat, height=None, distance=None)
#     sorted_indices = np.argsort(tphat[locs])[::-1]
#     pks = tphat[locs][sorted_indices]
#     locs = locs[sorted_indices]
#     print(locs[0:10])
#     td=locs[0]*1/FS
#     theta=np.degrees(np.arccos(343.3*td/6/d))%360-90
#     # print(theta)
#     time = np.asmatrix(np.arange(0,num_channel)*d*np.sin(np.degrees(theta))/C)


    
#     frame_count = frame_count + 1;
#     i = i + frame_shift;
   
#     print(i)








N = 31000; 

Fs = 48000; 

Ts = 1/Fs; #

Ndelay = 100;
rsig = np.random.randn(N*2, 1)
 

x = pcm[0:N].T[0];  
print(x.shape)
y=pcm[0:N].T[7]
print(y.shape)
X = np.fft.fft(x,axis=0);

Y = np.fft.fft(y,axis=0);

R = np.multiply(X,np.conj(Y));

tphat = np.real(np.fft.ifft(np.divide(R,np.abs(R)),axis=0));
tphat=np.reshape(tphat,(-1))
locs, _ = signal.find_peaks(tphat, height=None, distance=None)
sorted_indices = np.argsort(tphat[locs])[::-1]
pks = tphat[locs][sorted_indices]
locs = locs[sorted_indices]
print(locs)