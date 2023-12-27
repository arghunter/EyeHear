import numpy as np
from scipy import signal
from scipy.io.wavfile import read
class Signal: #delay in microseconds
    def __init__ (self, data, samplerate,delay):
        self.data=data
        self.samplerate=samplerate
        delay=np.zeros(int(delay/(1/samplerate*10**6)))
        self.data=np.reshape(np.concatenate([delay,self.data]),(-1,1))
    def sum(self, signal):
       l=min(self.data.shape[0],signal.data.shape[0])
       for i in range(l):
           self.data[i][0]+=signal.data[i][0] 
   
class Sine(Signal):
    
    def __init__(self, samplerate, delay,amp=1,length=1,period=1):
        data=amp* np.sin(2*np.pi*period*samplerate*np.linspace(0, length, length * samplerate, dtype=np.float32)[0:4800])
        super().__init__(data, samplerate, delay)

class Square(Signal):
    def __init__(self, samplerate, delay,amp=1,length=1,period=1):
        data=amp * signal.square(2*np.pi*period*samplerate*np.linspace(0, length, length * samplerate, dtype=np.float32)[0:4800])
        super().__init__(data, samplerate, delay)
        
                
class Wav(Signal):
    def __init__(self, filename, delay,amp=1,channel=0):
        file = read(filename)
        data =amp* np.array(file[1])[channel]
        samplerate=file[0]
        super().__init__(data, samplerate, delay)
        
class Sawtooth(Signal):
    def __init__(self, samplerate, delay,amp=1,length=1,period=1):
        data=amp * signal.sawtooth(2*np.pi*period*samplerate*np.linspace(0, length, length * samplerate, dtype=np.float32))
        super().__init__(data, samplerate, delay)
        
