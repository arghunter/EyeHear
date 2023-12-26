import numpy as np
from scipy.io.wavfile import write
class SignalSplicer:
    def __init__(self,samplerate):
        
        self.iter=0
        self.samplerate=samplerate
    
    def add_signal(self, signal):
        # print(signal.data.shape)
        if hasattr(self,'data'):
            self.data=np.concatenate([self.data,signal.data],axis=1)
        else:
            self.data=signal.data

        
    def write(self,fileName):
        write(fileName, self.samplerate, self.data)