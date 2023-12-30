import numpy as np
from scipy import signal
from IOStream import IOStream
from Preprocessor import Preprocessor
from VAD import VAD
v=340.3
class CrossCorrelatior: #spacing in meters,  lag =a of sample segments towait before correlation 
    def __init__(self,sample_rate=48000,n_channels=8,spacing=0.03,lag=3,past_buffer_length=30):
        self.sample_rate=sample_rate
        self.n_channels=n_channels
        self.spacing=spacing
        self.buffer=[0,0,0] # perhapsimplement 3+ samples
        self.buffer_head=0
        self.ticker=0
        self.lag=lag
        self.sample_dur= 1/sample_rate *10**6#Duration of a sample in microseconds
        self.past_buffer_length=int(past_buffer_length/lag)
        self.past_shift_buffer=np.zeros((self.past_buffer_length,n_channels))
        self.past_buffer_iter=0
        self.doa=0
    def get_doa(self,samples,signal_samp):
        # cycle buffer, if the middle sample is speech and half a second is lag then update the samples in a separate thread
        self.cycle_buffer(samples)
        if not signal_samp:
            self.ticker=self.lag-len(self.buffer)
        if self.ticker>=self.lag:
            self.ticker=0
            self.calculate_doa()
        else:
            self.ticker+=1
        return self.doa
    def calculate_doa(self):
        channel_shifts=np.zeros(self.n_channels)
        
        for i in range (1, self.n_channels):
            
            window1 = signal.windows.kaiser(len(self.buffer[1].T[0])*3,14)
            x=window1*np.concatenate([(self.buffer[0]).T[0],(self.buffer[1]).T[0],(self.buffer[2]).T[0]])
        
            # x=((self.buffer[1]).T[0]) *window
            # window2 = signal.windows.cosine(len(self.buffer[1].T[0]))
            y=window1*np.concatenate([np.zeros(480),(self.buffer[1]).T[i],np.zeros(480)])

            # y=((self.buffer[1]).T[i]) *window2
        
            cross_corr=signal.correlate(x,y)
            lags=signal.correlation_lags(len(x),len(y))
            #TODO: add some weights here
            lag=-lags[np.argmax(cross_corr)]
            channel_shifts[i]=lag
        if self.past_buffer_iter>=self.past_buffer_length:
            self.past_buffer_iter=-1
        
        self.past_shift_buffer[self.past_buffer_iter]=channel_shifts
        self.past_buffer_iter+=1
        # for i in range(1,self.n_channels):
            
        #     diff = np.abs(self.past_shift_buffer.T[i] - np.median(self.past_shift_buffer.T[i]))
        #     dev = np.median(diff)
        #     s = diff/dev if dev else np.zeros(len(diff))
        #     arr=self.past_shift_buffer.T[i][s<2]
        #     if channel_shifts[i]>max(arr) or channel_shifts[i]>min(arr):
        #         channel_shifts[i]=self.channel_shifts[i]
                
        # self.channel_shifts=channel_shifts
            
        
        # print("Cross-correlation result:", cross_corr)
        # print("Corresponding lags:", lags)
        # print("Lag:", lag)
        self.doa=self.shift_to_angle(channel_shifts)
        
    def cycle_buffer(self,samples):
        if self.buffer_head<len(self.buffer):
            self.buffer[self.buffer_head]=samples
            self.buffer_head+=1
        else:    
            self.buffer_head=0
            self.buffer[self.buffer_head]=samples
    def shift_to_angle(self,channel_shifts):
        delays=channel_shifts*self.sample_dur
        ang=np.zeros(self.n_channels)
        for i in range(1,self.n_channels):
            ang[i]=np.degrees(np.arccos(v*delays[i]/i/self.spacing/(10**6)))%360
        return np.mean(ang[~np.isnan(ang)])
    def update_delays(self,doa): #doa in degrees, assuming plane wave as it is a far-field source
        for i in range(self.n_channels):
            self.delays[i]=(i*self.spacing*np.cos(np.radians(doa))/v)*10**6
            # v*self.delays[i]/((10**6)*self.spacing*i)
        shift=min(self.delays)
        self.delays+=-shift
       
  
  
# io=IOStream()
# io.wavToStream("./beamformingarray/output5_000v2.wav")
# pre=Preprocessor()
# cross=CrossCorrelatior()
# for i in range(100):
#     io.getNextSample()
# for i in range(100):
#     print(cross.get_doa(pre.process(io.getNextSample()),True))
#     print(i)
# def reject_outliers(data, m = 2.):
#     d = np.abs(data - np.median(data))
#     mdev = np.median(d)
#     s = d/mdev if mdev else np.zeros(len(d))
#     return data[s<m]

# arr = np.array([  -29,  -29,  -29,  -29,  -34,  -29,  -256,  -29,  -29,  -30,  -25,  -27,  90,  -28,  -29,  -31,  -16,  -30,  -30,  54,  104,  -95,  5,  2,  -29]) 
# print(reject_outliers(arr)) 
# io=IOStream()
# io.wavToStream("./beamformingarray/output5_100v2.wav")
# pre=Preprocessor()
# cross=CrossCorrelatior()
# for i in range(140):
#     io.getNextSample()
# for i in range(100):
#     cross.get_channel_shifts(pre.process(io.getNextSample()))
#     print(i)   
# # Two example signals
# signal_1 = np.array([0, 1,0,1, 0,1])
# signal_2 = np.array([1, 0,1, 0, 1,0])

# # Perform cross-correlation
# cross_corr = correlate(signal_1, signal_2)

# # Get the lags corresponding to the cross-correlation
# lags = correlation_lags(len(signal_1), len(signal_2))
# lag = lags[np.argmax(cross_corr)]
# print("Cross-correlation result:", cross_corr)
# print("Corresponding lags:", lags)
# print("Lag:", lag)