import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf
from scipy import fft
from Signal import Sine
from SignalGen import SignalGen
from IOStream import IOStream
from AudioWriter import AudioWriter
from scipy.io.wavfile import read
# io= IOStream(sample_duration=20000)
# io.wavToStream("./beamformingarray/AudioTests/test_input_sig.wav")
file = read("./beamformingarray/AudioTests/test_input_sig.wav")
pcm=np.array(file[1])
FS=file[0]
N=pcm.shape[0]
stft_len=1024
num_channel=pcm.shape[1]
frame_len=960
frame_shift=480 # 50% overlap
exp_avg_param=50
win=signal.windows.hann(frame_len)
win=win*frame_shift/sum(win) # why?
win_multi=np.tile(win,(num_channel,1)).T
frame_num=np.floor((N-frame_len)/frame_shift+1);
output=np.zeros((N,1))
N_f=int(stft_len/2+1)#Num frequencies
global_covar=np.zeros((num_channel,num_channel,N_f))
frame_count=1
j=1
mu=0
while j + frame_len < N:
    print(j)
    win_data = pcm[j : j + frame_len,:]*win_multi
    spectrum=fft.fft(win_data,stft_len)
    if frame_count < exp_avg_param:
        mu=(frame_count-1)/frame_count
    for k in range(1,N_f):
        cov_mat=np.dot(spectrum[k,:].T,np.conj((spectrum[k,:])))
        corr_mat=cov_mat/np.trace(cov_mat); 
        global_covar[:,:,k]=mu*global_covar[:,:,k]+(1-mu)*corr_mat
    time=np.zeros(1,num_channel)
    w=np.zeros(num_channel,N_f)
    for k in range (0,N_f-1):
        f=k*FS/stft_len;
        alpha=np.exp(-j*2*np.pi*f*time).T
        r_inv=np.linalg.pinv(global_covar[:,:,k+1]+(1e-8)*np.diag(np.ones((num_channel,1))))
        w[:,k+1]=r_inv*alpha/(np.conj(alpha.T)*r_inv*alpha)
    rec_signal=np.conj(w.T)*spectrum[1:N_f,:]# check line
    rec_signal=[rec_signal]
    #### CHECK THIS
    submatrix = rec_signal[1:-1, :]
    flipped_conjugate = np.flipud(np.conj(submatrix))
    rec_signal = np.vstack([rec_signal, flipped_conjugate])
    ####    
    res=np.real(np.fft.ifft(np.sum(rec_signal,axis=1)))
    res=res[1:frame_len]
    output[j:j+frame_len-1,:]=output[j:j+frame_len-1,:]+res
    
    frame_count = frame_count + 1; 

    j = j + frame_shift;
    
    
aw=AudioWriter()
aw.add_sample(output)
aw.write("./beamformingarray/AudioTests/8.wav",48000)