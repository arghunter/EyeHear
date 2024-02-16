import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
v=343.3
class MUSIC:
    
    def __init__(self,sample_rate=48000,spacing=np.array([0,0.028,0.056,0.084,0.112,0.14,0.168,0.196]),num_channels=8,frame_len=960,stft_len=1024,nsrc=3,acc=10):
        self.spacing=spacing
        self.num_channels=num_channels

        self.frame_len=frame_len
        self.stft_len=stft_len
        self.stftd2=int(stft_len/2)
        self.sample_rate=sample_rate
        self.frame_shift=int(frame_len/2)
        self.N_f=int(stft_len/2)+1
        self.acc=acc
       
       
        self.angles = np.arange(-90, 91, 1)
        self.Ng = len(self.angles)
        self.nsrc = nsrc
        self.sources=np.zeros(nsrc)
        self.weights=np.ones(nsrc)
    def doa(self,global_covar):
        import numpy as np

        # Initialize spatial spectrum array
        w = np.zeros(self.Ng)

        # Iterate over frequency bins
        for k in range(self.N_f):
            f = k * self.sample_rate / self.stft_len  
            Rxx = global_covar[:, :, k] 
            U, evi, _ = np.linalg.svd(Rxx)  
            eig_vals = np.diag(evi)
            En = U[:, self.nsrc:]  

        
            for g in range(self.Ng):
               
                c_theta = self.angles[g]
                c_time = self.spacing * np.sin(np.deg2rad(c_theta)) / v
                c_alpha = np.exp(-1j * 2 * np.pi * f * c_time)
                np_val = np.abs(c_alpha.conj().T @ En @ En.conj().T @ c_alpha)
                power = 1 / np_val
                w[g] += power
        plt.plot(self.angles,w)
        plt.show()
        locs, _ = signal.find_peaks(w, height=None, distance=None)
        sorted_indices = np.argsort(w[locs])[::-1]
        pks = w[locs][sorted_indices]
        locs = locs[sorted_indices]
        print(locs[0:10])
        print(pks[0:10])
        return locs[0]
    def source_tracker(self, locs,pks):
        if True or self.weights[0]==0:
            count=0
            
            for i in range(len(locs)):
                shift=False
                if count<self.nsrc:
                    while count<self.nsrc and (locs[i]-self.sources[count])>self.acc:
                        count+=1
                        shift=True
                    if count==self.nsrc:
                        count-=1
                    if shift:
                        self.sources[count]=locs[i]
                        self.weights[count]=pks[i]
                        
                    else:
                        self.sources[count]= self.weights[count]*self.sources[count]+locs[i]*pks[i]/(pks[i]+self.weights[count])
                        self.weights[count]+=pks[i]
        else:
            pass
            # Decrease by percent then samething and sorting
        
