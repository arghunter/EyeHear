from scipy import signal
from SignalSplicer import SignalSplicer
import numpy as np
from time import time
from Signal import Sawtooth,Sine,Square,Wav
from scipy.io.wavfile import read
from IOStream import IOStream
# saw=Sawtooth(48000,100000,period=1)P
# trans=saw.data.T
# splicer =SignalSplicer(48000)
# sine=Sine(48000,0,period=6,length=10)
# splicer.add_signal(sine)
# sine2=Sine(48000,1000000,period=6,length=10)
file = read("./beamformingarray/output5_100.wav")
arr = np.array(file[1])
channels=len(file[1][0])
frequency=file[0]

io=IOStream()
io.wavToStream("./beamformingarray/output5_100.wav")
for i in range(1000):
    samp=io.getNextSample()
    x = samp.T[0]
    y = samp.T[14]
    t1=int(time() * 1000)
    correlation = signal.correlate(x, y)
    # print(correlation)
    lags = signal.correlation_lags(x.size, y.size)
    lag = lags[np.argmax(correlation)]
    # print(int(time() * 1000)-t1)
    print(lag)
    # print(type(io.getNextSample()))
    

# sine=Square(48000,0,period=6,length=1)
# splicer.add_signal(sine)
# sine=Sawtooth(48000,0,period=1,length=1)
# splicer.add_signal(sine)
# splicer.write("./beamformingarray/splicetest1.wav")    
rng = np.random.default_rng()
x = arr.T[0]
y = arr.T[2]
t1=int(time() * 1000)
correlation = signal.correlate(x, y, mode="full")
print(correlation)
lags = signal.correlation_lags(x.size, y.size, mode="full")
lag = lags[np.argmax(correlation)]
print(int(time() * 1000)-t1)
print(lag)