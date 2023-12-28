
import numpy as np
from AudioWriter import AudioWriter
from IOStream import IOStream
from Preprocessor import Preprocessor
from time import time
import sounddevice as sd
from scipy import signal

v=340.3 # speed of sound at sea level m/s
    
class Beamformer:
    def __init__(self,n_channels=8,spacing=0.03,sample_rate=48000):
        self.n_channels = n_channels
        self.spacing = spacing
        self.sample_rate = sample_rate
        self.delays = np.zeros(n_channels) #in microseconds
        self.gains = np.ones(n_channels) # multiplier
        self.sample_dur= 1/sample_rate *10**6 #Duration of a sample in microseconds
        
    def beamform(self,samples):
        # print(samples.shape[0])
        
        samples,max_sample_shift=self.delay_and_gain(samples)
        # print(max_sample_shift)
        samples=self.sum_channels(samples)
        if hasattr(self,'last_overlap'):
            for i in range(self.last_overlap.shape[0]):
                samples[i]+=self.last_overlap[i]
        
        self.last_overlap=samples[samples.shape[0]-max_sample_shift:samples.shape[0]]
        # print(self.last_overlap.shape[0])
        return samples[0:samples.shape[0]-max_sample_shift]
    
    def sum_channels(self,samples):
        summed=np.zeros(samples.shape[0])
        for j in range(samples.shape[0]):
            summed[j] = samples[j].sum()
        return summed
    def delay_and_gain(self, samples):
        #backwards interpolations solves every prblem
        shifts=self.calculate_channel_shift(samples)
        intshifts=np.floor(shifts)
        max_sample_shift=int(max(intshifts))
        dims = samples.shape
        dims=(int(dims[0]+max_sample_shift),dims[1])
        delayed = np.zeros(dims)
        if hasattr(self,'last_samples'):
            
            for i in range(self.n_channels):
                intermult=1-(shifts[i]%1)
                shiftdiff=max_sample_shift-int(intshifts[i])
                delayed[0+shiftdiff][i]=self.gains[i]*((samples[0][i]-self.last_samples[len(self.last_samples)-1][i])*(intermult)+self.last_samples[len(self.last_samples)-1][i])               
        else:
            for i in range(self.n_channels):
                intermult=1-(shifts[i]%1)
                shiftdiff=max_sample_shift-int(intshifts[i])
                delayed[0+shiftdiff][i]=(self.gains[i]*(samples[0][i]-0)*(intermult))               
        
        for i in range(self.n_channels):
            intermult=1-(shifts[i]%1)
            shiftdiff=max_sample_shift-int(intshifts[i])
            for j in range(1,dims[0]-max_sample_shift):
                delayed[j+shiftdiff][i]=self.gains[i]*((samples[j][i]-samples[j-1][i])*(intermult)+samples[j-1][i])               
            
        
        self.last_samples=samples
       
        return delayed,max_sample_shift
    #calculates number of samples to delay
    def calculate_channel_shift(self,samples):
        # # really should interpolate
        # transpose=samples.T
        
        # channel_shifts=np.zeros(self.n_channels)
        # for i in range(int(self.n_channels/2),self.n_channels):
        #     x = transpose[0]
        #     y = transpose[i]
            
            
        #     # t1=int(time() * 1000)
        #     correlation = signal.correlate(x, y)
        #     # print(correlation)
        #     lags = signal.correlation_lags(x.size, y.size)
        #     channel_shifts[i] = max(lags[np.argmax(correlation)],0)
        #     # channel_shifts[i]=signal.correlation_lags(transpose[0].size, transpose[i].size)[signal.correlate(transpose[0],transpose[i])]
        # for i in range(1,int(self.n_channels/2)):
        #     x = transpose[i]
        #     y = transpose[self.n_channels-1]
            
        #     # t1=int(time() * 1000)
        #     correlation = signal.correlate(x, y)
        #     # print(correlation)
        #     lags = signal.correlation_lags(x.size, y.size)
        #     channel_shifts[i] = max(0,channel_shifts[self.n_channels-1]-lags[np.argmax(correlation)])
        #     # channel_shifts[i]=channel_shifts[self.n_channels-1] - signal.correlation_lags(transpose[i].size, transpose[self.n_channels-1].size)[signal.correlate(transpose[i],transpose[self.n_channels-1])]
        # print(channel_shifts)
        # return channel_shifts
        channel_shifts=(self.delays/self.sample_dur)
        print(channel_shifts)
        return channel_shifts

    def update_delays(self,doa): #doa in degrees, assuming plane wave as it is a far-field source
        for i in range(self.n_channels):
            self.delays[i]=(i*self.spacing*np.cos(np.radians(doa))/v)*10**6
        shift=min(self.delays)
        self.delays+=-shift
    def update_gains(self,doa):
        pass
    
    

beam = Beamformer(8,spacing=0.03,sample_rate=1*48000)
beam.update_delays(00)
io=IOStream()
writer= AudioWriter()
io.wavToStream("./beamformingarray/output5_100.wav")
sd.default.device=20
# io= IOStream()
# io.streamAudio(48000,16)
pre=Preprocessor()

for i in range(1000):
    t1=int(time() * 1000)
    writer.add_sample(beam.beamform((pre.process(io.getNextSample()))))
    # writer.add_sample(io.getNextSample())
    print(int(time() * 1000)-t1)
    # print(type(io.getNextSample()))
    
writer.write("./beamformingarray/output5_100res4.wav",1*48000)   
# print(res)
# beam = Beamformer(4)
# beam.update_delays(80)
# arr=np.zeros((20,4))
# for i in range (20):
#     arr[i][0]=i+1
#     arr[i][1]=i+1
#     arr[i][2]=i+1
#     arr[i][3]=i+1
# res,temp=beam.beamform(arr)
# print(res)