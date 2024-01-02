
import numpy as np
from AudioWriter import AudioWriter
from IOStream import IOStream
from Preprocessor import Preprocessor
from time import time
import sounddevice as sd
from scipy import signal
from CrossCorrelator import CrossCorrelatior
from VAD import VAD
v=340.3 # speed of sound at sea level m/s
    
class Beamformer:
    def __init__(self,n_channels=8,spacing=0.03,sample_rate=48000):
        self.n_channels = n_channels
        self.spacing = spacing
        self.sample_rate = sample_rate
        self.delays = np.zeros(n_channels) #in microseconds
        self.gains = np.ones(n_channels) # multiplier
        self.sample_dur= 1/sample_rate *10**6 #Duration of a sample in microseconds
        self.cc=CrossCorrelatior(self.sample_rate,self.n_channels,self.spacing,lag=4)
        self.vad=VAD()
    def beamform(self,samples):
        # print(samples.shape[0])
        sample_save=samples
        samples,max_sample_shift=self.delay_and_gain(samples)
        # print(max_sample_shift)
        samples=self.sum_channels(samples)
        if hasattr(self,'last_overlap'):
            for i in range(self.last_overlap.shape[0]):
                samples[i]+=self.last_overlap[i]
        
        self.last_overlap=samples[samples.shape[0]-max_sample_shift:samples.shape[0]]
        speech=True
        # speech=self.vad.is_speech(samples[0:samples.shape[0]-max_sample_shift])
        self.update_delays(self.cc.get_doa(sample_save,speech))
        return samples[0:samples.shape[0]-max_sample_shift]
    
    def sum_channels(self,samples):
        summed=np.zeros(samples.shape[0])
        for j in range(samples.shape[0]):
            summed[j] = samples[j].sum()
        return summed
    def delay_and_gain(self, samples):
        #backwards interpolations solves every prblem
        shifts=self.calculate_channel_shift()
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
    def calculate_channel_shift(self):
        channel_shifts=(self.delays/self.sample_dur)
        return channel_shifts

    def update_delays(self,doa): #doa in degrees, assuming plane wave as it is a far-field source
        print(doa)
        for i in range(self.n_channels):
            self.delays[i]=(i*self.spacing*np.cos(np.radians(doa))/v)*10**6
        shift=min(self.delays)
        self.delays+=-shift
    def update_gains(self,doa):
        pass
    
    

# beam = Beamformer(8,spacing=0.03,sample_rate=1*48000)
# beam.update_delays(0)
# io=IOStream()
# writer= AudioWriter()
# io.wavToStream("./beamformingarray/gentest17.wav")
# # sd.default.device=20
# # io= IOStream()
# # io.streamAudio(48000,16)
# pre=Preprocessor(mirrored=False)

# for i in range(100):
#     t1=int(time() * 1000)
#     writer.add_sample(beam.beamform((pre.process(io.getNextSample()))))
#     # writer.add_sample(io.getNextSample())
#     print(int(time() * 1000)-t1)
#     # print(type(io.getNextSample()))
    
# writer.write("./beamformingarray/gentest17res1.wav",1*48000)   
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