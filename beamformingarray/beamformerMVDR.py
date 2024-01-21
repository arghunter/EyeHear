import numpy as np
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import scipy.signal as signal
from IOStream import IOStream
from AudioWriter import AudioWriter

C=343.3
class Beamformer():
    
    def __init__(self,sample_rate=48000,spacing=0.028,num_channels=8,exp_avg=50,frame_len=960,stft_len=1024):
        self.spacing=spacing
        self.num_channels=num_channels
        self.exp_avg=exp_avg
        self.frame_count=1
        self.frame_len=frame_len
        self.stft_len=stft_len
        self.sample_rate=sample_rate
        self.frame_shift=int(frame_len/2)
        win=np.hanning(frame_len)
        win=win*self.frame_shift/sum(win)
        self.win_multi=np.tile(win,(self.num_channels,1)).T
        self.N_f=int(stft_len/2)+1
        self.mu=0
        self.global_covar=np.zeros((num_channels,num_channels,self.N_f),dtype='complex128')
    def beamform(self,frame):
        if(len(frame)!=self.frame_len):
            return np.zeros((self.frame_len,1))
        # print(pcm[j : j + frame_len,:].shape)
        win_data = np.asmatrix(frame*self.win_multi)
        
        
        spectrum=np.asmatrix(np.fft.rfft(win_data,self.stft_len,axis=0))
        # print(spectrum.shape)
        if self.frame_count < self.exp_avg:
            self.mu=(self.frame_count-1)/self.frame_count
        for k in range(0,self.N_f):
            cov_mat=np.dot(spectrum[k,:].T, np.conj(spectrum[k,:]))
            corr_mat=cov_mat/np.trace(cov_mat); 
            # print(corr_mat.dtype)
            # print(cov_mat.shape)
            
            self.global_covar[:, :, k] = self.mu * self.global_covar[:, :, k] + (1 - self.mu) * corr_mat
        # print(win_data.T[0].T.shape)
        # x=pcm[i : i + frame_len,:].T[0].T
        # y=pcm[i : i + frame_len,:].T[5].T    
        # cross_corr=signal.correlate(x,y)
        # lags=signal.correlation_lags(len(x),len(y))
        # lag=lags[np.argmax(cross_corr)]
        # theta=np.degrees(np.arccos(343.3*lag/6/d))%360-90
        # print(lag)
        theta=0
        time = np.asmatrix(np.arange(0,self.num_channels)*self.spacing*np.sin(np.degrees(theta))/C)
        w=np.asmatrix(np.zeros((self.num_channels,self.N_f),dtype='complex128'))
        for k in range (0,self.N_f-1):
            f=k*self.sample_rate/self.stft_len;
            alpha=np.exp(-1j*2*np.pi*f*time).T
            r_inv=np.linalg.pinv(self.global_covar[:,:,k]+(1e-8)*np.eye(self.num_channels)) # this is bad
            w[:,k]=r_inv@alpha/(np.conj(alpha.T)@r_inv@alpha)
        rec_signal=np.multiply(w.H,spectrum[0:self.N_f,:])# Works till here the next block may be an issue

        submatrix = rec_signal[1:-1, :]

        flipped_conjugate = np.flipud(np.conj(submatrix))
        rec_signal = np.vstack([rec_signal, flipped_conjugate])


        summed_signal=(np.sum(rec_signal,axis=1))
        
        res_comp=(np.fft.ifft(summed_signal, axis=0))
        res=np.real(res_comp)
        
        res=res[0:self.frame_len]
        # print((output[i:i + frame_len, :]).shape)
        self.frame_count+=1
        return res
        

io=IOStream()
aw=AudioWriter()
file = read("./beamformingarray/AudioTests/test_input_sig.wav")
beam=Beamformer()
pcm=np.array(file[1])/32767
io.arrToStream(pcm,48000)
while(not io.complete()):
    aw.add_sample(beam.beamform(io.getNextSample()),480)
aw.write("./beamformingarray/AudioTests/10.wav",48000)