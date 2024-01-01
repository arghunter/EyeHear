import numpy as np
from Signal import Sine,Sawtooth,Square,Signal,Chirp
import Signal
from SignalSplicer import SignalSplicer
from AudioWriter import AudioWriter
v=340.3



# def angleToWav(angles,signals,n_mics,spacing,samplerate=48000):
#     for i in range(len(signals)):
#         signal=signals[i]
#         doa=angles[i]
        
        
class SignalGen:
    def __init__(self,n_channels=8,spacing=0.03,sample_rate=48000):
        self.n_channels = n_channels
        self.spacing = spacing
        self.sample_rate = sample_rate
        self.delays = np.zeros(n_channels) #in microseconds
        self.gains = np.ones(n_channels) # multiplier
        self.sample_dur= 1/sample_rate *10**6 #Duration of a sample in microseconds
   
    

    def delay_and_gain(self, samples):
        
        shifts=self.calculate_channel_shift()
       
        max_sample_shift=int(max(shifts))
        dims = samples.shape
        dims=(int(dims[0]+max_sample_shift),dims[1])
        delayed = np.zeros((dims[0],self.n_channels))
        for i in range(self.n_channels):
            for j in range(dims[0]-max_sample_shift):
                delayed[j+int(shifts[i])][i]=self.gains[i]*samples[j][0]               
            
        
        return delayed
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


    
    
# splicer =SignalSplicer(48000)
gen=SignalGen(spacing=0.03)
writer= AudioWriter()
sine=Chirp(start_freq=2,end_freq=100)
sine_data=sine.generate_wave(1)
saw=Sawtooth(frequency=4,amplitude=0.4)
saw_data=saw.generate_wave(0.6)
gen.update_delays(0)
sine_data=gen.delay_and_gain(sine_data)
gen.update_delays(135)
saw_data=gen.delay_and_gain(saw_data)
sum_data = Signal.sum_signals(saw_data,sine_data)
# print(sum_data.shape)
sum_data=Signal.add_noise(sum_data,noise_level=0.2)
# print(sum_data.shape)
writer.add_sample(sum_data)
writer.write("./beamformingarray/gentest16.wav",48000)
# sine.sum(Sawtooth(48000,0,period=6)) # Sum
# gen.update_delays(0)
# sig1=Signal(gen.delay_and_gain(sine.data),48000,0,t=2)
# gen.update_delays(90)
# saw=Sawtooth(48000,0,period=1,amp=0.5)
# sig1.sum(Signal(gen.delay_and_gain(saw.data),48000,0,t=2))
# writer.add_sample(sig1.data)
# writer.write("./beamformingarray/gentest14.wav",48000) 
# writer2=AudioWriter()
# writer2.add_sample(sine.data)
# writer2.write("./beamformingarray/gentest14o.wav",48000)
# saw=Sawtooth(48000,100000,period=1)
# splicer =SignalSplicer(48000)
# sine=Sawtooth(48000,0,period=2,length=1)
# splicer.add_signal(sine)
# sine=Sine(48000,0,period=6,length=1)
# splicer.add_signal(sine)
# sine=Square(48000,0,period=6,length=1)
# splicer.add_signal(sine)
# sine=Sawtooth(48000,0,period=1,length=1)
# splicer.add_signal(sine)
# splicer.write("./beamformingarray/splicetest1.wav")