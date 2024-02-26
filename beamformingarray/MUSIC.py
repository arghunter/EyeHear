import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from DelayApproximation import DelayAproximator
v=343.3
class MUSIC:
    
    def __init__(self,sample_rate=48000,spacing=np.array([[0,0],[0.028,0],[0.056,0],[0.084,0],[0.112,0],[0.14,0],[0.168,0],[0.196,0]]),num_channels=8,frame_len=960,stft_len=1024,nsrc=3,acc=10,decay=0.25,saturation=8000):
        self.spacing=spacing
        self.num_channels=num_channels

        self.frame_len=frame_len
        self.stft_len=stft_len
        self.stftd2=int(stft_len/2)
        self.sample_rate=sample_rate
        self.frame_shift=int(frame_len/2)
        self.N_f=int(stft_len/2)+1
        self.acc=acc
        self.decay=decay
       
        # self.angles = np.arange(-180, 181, 1)
        self.angles = np.arange(-90, 91, 1)
        self.Ng = len(self.angles)
        self.nsrc = nsrc
        self.sources=(np.arange(nsrc)*270/self.nsrc) 
        # print(self.sources)
        self.weights=np.ones(nsrc)
        self.saturation=saturation
        self.delay_approx=DelayAproximator(self.spacing)
    def doa(self,global_covar):
        # print("Here")
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
                # c_time = np.array(self.delay_approx.get_delays(DelayAproximator.get_pos(c_theta+180,2)))
                c_time = np.array(self.delay_approx.get_delays(DelayAproximator.get_pos(c_theta+90,2)))
                c_alpha = np.exp(-1j * 2 * np.pi * f * c_time)
                np_val = np.abs(c_alpha.conj().T @ En @ En.conj().T @ c_alpha)
                power = 1 / np_val
                w[g] += power
        # plt.plot(self.angles,w)
        # plt.show()
        locs, _ = signal.find_peaks(w, height=200, distance=10)
        sorted_indices = np.argsort(w[locs])[::-1]
        pks = w[locs][sorted_indices]
        locs = locs[sorted_indices]
        self.source_tracker(locs,pks)
        # print(locs)
        # print(pks)
        # print(self.sources)
        # print(self.weights)

        return self.sources[self.nsrc-1]
    def source_tracker(self, locs,pks):
        if  False and self.weights[0]==1:
            count=0
            
            for i in range(len(locs)):
                shift=False
                if count<self.nsrc:
                    while count<self.nsrc and (np.abs(locs[i]-self.sources[count])>self.acc and not self.sources[count]==0.0):
                        
                        count+=1
                        shift=True
                    
                    if count==self.nsrc:
                        count-=1
                    if shift:
                        self.sources[count]=locs[i]
                        self.weights[count]=pks[i]
                        
                    else:
                        self.sources[count]= (self.weights[count]*self.sources[count]+locs[i]*pks[i])/(pks[i]+self.weights[count])
                        self.weights[count]+=pks[i]
                        self.weights[count]= min(self.weights[count],self.saturation)
        else:
            self.weights*=(1-self.decay)
            for i in range(len(locs)):
                
                min_dif=100000000
                min_source=0
                for j in range(self.nsrc):
                    if self.sources[j]%1==0:
                        min_source=j
                        break
                        
                    if  np.abs(locs[i]-self.sources[j])<min_dif :
                        min_dif=np.abs(locs[i]-self.sources[j])
                        min_source=j
                self.sources[min_source]= (self.weights[min_source]*self.sources[min_source]+locs[i]*pks[i])/(pks[i]+self.weights[min_source])

                self.weights[min_source]+=pks[i]
                # if i < len(locs):
                #     self.sources[self.nsrc-1-i]= (self.weights[self.nsrc-1-i]*self.sources[self.nsrc-1-i]+locs[i]*pks[i])/(pks[i]+self.weights[self.nsrc-1-i])

                #     self.weights[self.nsrc-1-i]+=pks[i]

        ind=np.argsort(self.weights)
        self.weights=np.sort(self.weights)
        self.sources=self.sources[ind]

            # Decrease by percent then samething and sorting
        
