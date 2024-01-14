import numpy as np
from scipy.io.wavfile import read
from scipy.io.wavfile import write
# io= IOStream(sample_duration=20000)
# io.wavToStream("./beamformingarray/AudioTests/test_input_sig.wav")
file = read("./beamformingarray/AudioTests/test_input_sig_1.wav")

pcm=np.array(file[1])/32767
# print(np.max(pcm))
FS=file[0]
N=pcm.shape[0]
stft_len=1024
num_channel=pcm.shape[1]
frame_len=960
frame_shift=480 # 50% overlap
exp_avg_param=50
win=np.hanning(frame_len)
win=win*frame_shift/sum(win) # why?
win_multi=np.tile(win,(num_channel,1)).T
frame_num=int(np.floor((N-frame_len)/frame_shift+1));
# print(frame_num)
# N=31000
output=np.zeros((N,1))
N_f=int(stft_len/2)+1#Num frequencies
global_covar=np.zeros((num_channel,num_channel,N_f),dtype='complex128')
frame_count=1
i=0
mu=0

while i + frame_len < N:# while i<=0:# 
    # print(pcm[j : j + frame_len,:].shape)
    win_data = np.asmatrix(pcm[i : i + frame_len,:]*win_multi)
    # print(win_data)
    spectrum=np.asmatrix(np.fft.fft(win_data,stft_len,axis=0))
    # print(spectrum.shape)
    if frame_count < exp_avg_param:
        mu=(frame_count-1)/frame_count
    for k in range(0,N_f):
        cov_mat=np.dot(spectrum[k,:].T, np.conj(spectrum[k,:]))
        corr_mat=cov_mat/np.trace(cov_mat); 
        # print(corr_mat.dtype)
        # print(cov_mat.shape)
        
        global_covar[:, :, k] = mu * global_covar[:, :, k] + (1 - mu) * corr_mat
    # print(i)
    # print(global_covar.dtype)
    time=np.asmatrix(np.zeros((1,num_channel)))
    w=np.asmatrix(np.zeros((num_channel,N_f),dtype='complex128'))
    for k in range (0,N_f-1):
        f=k*FS/stft_len;
        alpha=np.exp(-1j*2*np.pi*f*time).T
        # print(alpha.dtype)
        r_inv=np.linalg.pinv(global_covar[:,:,k]+(1e-8)*np.eye(num_channel)) # this is bad
        w[:,k]=r_inv@alpha/(np.conj(alpha.T)@r_inv@alpha)
    rec_signal=np.multiply(w.H,spectrum[0:N_f,:])# Works till here the next block may be an issue
    # rec_signal=[rec_signal]
    # print(i)
    # print(rec_signal.shape)
    #### CHECK THIS
    submatrix = rec_signal[1:-1, :]
    # print(submatrix.shape)
    flipped_conjugate = np.flipud(np.conj(submatrix))
    rec_signal = np.vstack([rec_signal, flipped_conjugate])
    # print(rec_signal.shape)
    ####
    summed_signal=(np.sum(rec_signal,axis=1))
    
    res_comp=(np.fft.ifft(summed_signal, axis=0))
    res=np.real(res_comp)
    
    res=res[0:frame_len]
    # print(np.mean(res))
    output[i:i + frame_len, :] += res
    
    frame_count = frame_count + 1;
    i = i + frame_shift;
# plot=plt.plot(np.arange(0,len(res)),res)
# plot.show()  
write("./beamformingarray/AudioTests/8.wav", 48000, output)
