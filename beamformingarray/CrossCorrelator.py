import numpy as np
from scipy import signal
from IOStream import IOStream
from Preprocessor import Preprocessor
from VAD import VAD
class CrossCorrelatior: #spacing in meters,  lag =a of sample segments towait before correlation 
    def __init__(self,sample_rate=48000,n_channels=8,spacing=0.03,lag=3):
        self.sample_rate=sample_rate
        self.n_channels=n_channels
        self.spacing=spacing
        self.buffer=[0,0,0] # perhapsimplement 3+ samples
        self.buffer_head=0
        self.channel_shifts=np.zeros(self.n_channels)
        self.buffer_full=False
        self.ticker=0
        self.lag=lag
    def get_channel_shifts(self,samples):
        # cycle buffer, if the middle sample is speech and half a second is lag then update the samples in a separate thread
        self.cycle_buffer(samples)
        if self.ticker>=self.lag:
            self.ticker=0
            self.calculate_channel_shifts()
        else:
            self.ticker+=1
        return self.channel_shifts
    def calculate_channel_shifts(self):
        window1 = signal.windows.cosine(len(self.buffer[1].T[0])*3)
        x=window1*np.concatenate([(self.buffer[0]).T[0],(self.buffer[1]).T[0],(self.buffer[2]).T[0]])
        
        # x=((self.buffer[1]).T[0]) *window
        window2 = signal.windows.cosine(len(self.buffer[1].T[0]))
        y=window1*np.concatenate([(self.buffer[0]).T[self.n_channels-1],(self.buffer[1]).T[self.n_channels-1],(self.buffer[2]).T[self.n_channels-1]])

        # y=((self.buffer[1]).T[self.n_channels-1]) *window2
        
        cross_corr=signal.correlate(x,y)
        lags=signal.correlation_lags(len(x),len(y))
        lag=lags[np.argmax(cross_corr)]
        
        # print("Cross-correlation result:", cross_corr)
        # print("Corresponding lags:", lags)
        print("Lag:", lag)
        
    def cycle_buffer(self,samples):
        if self.buffer_head<len(self.buffer):
            self.buffer[self.buffer_head]=samples
            self.buffer_head+=1
        else:    
            self.buffer_head=0
            self.buffer[self.buffer_head]=samples
            
  
  
io=IOStream()
io.wavToStream("./beamformingarray/output5_000v2.wav")
pre=Preprocessor()
cross=CrossCorrelatior()
for i in range(100):
    io.getNextSample()
for i in range(100):
    cross.get_channel_shifts(pre.process(io.getNextSample()))
    print(i)
# def reject_outliers(data, m = 2.):
#     d = np.abs(data - np.median(data))
#     mdev = np.median(d)
#     s = d/mdev if mdev else np.zeros(len(d))
#     return data[s<m]

# arr = np.array([-29,-29,-29,-34,-29,-29,-28,-29,-30,-23,-26,89,-50,-6,-31,-16,-30,-30,27,78,-95,5,30,-29]) 
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