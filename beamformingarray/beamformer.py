
import numpy as np
from AudioWriter import AudioWriter
from IOStream import IOStream
from Preprocessor import Preprocessor
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
        
        shifts=self.calculate_channel_shift()
       
        max_sample_shift=int(max(shifts))
        dims = samples.shape
        dims=(int(dims[0]+max_sample_shift),dims[1])
        delayed = np.zeros(dims)
        for i in range(self.n_channels):
            for j in range(dims[0]-max_sample_shift):
                delayed[j+max_sample_shift-int(shifts[i])][i]=self.gains[i]*samples[j][i]               
            
        
        return delayed,max_sample_shift
    #calculates number of samples to delay
    def calculate_channel_shift(self):
        channel_shifts=np.around((self.delays/self.sample_dur))
        return channel_shifts
    def update_delays(self,doa): #doa in degrees, assuming plane wave as it is a far-field source
        for i in range(self.n_channels):
            self.delays[i]=(i*self.spacing*np.cos(np.radians(doa))/v)*10**6
        shift=min(self.delays)
        self.delays+=-shift
    def update_gains(self,doa):
        pass
    
    

beam = Beamformer(8)
# beam.update_delays(0)
io=IOStream()
writer= AudioWriter()
io.wavToStream("./beamformingarray/test1.wav")
pre=Preprocessor()
for i in range(300):
    writer.add_sample(beam.beamform(pre.process(io.getNextSample())))
    # print(type(io.getNextSample()))
    
writer.write("./beamformingarray/test1res2.wav",48000)   
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