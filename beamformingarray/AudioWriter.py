from scipy.io.wavfile import write
import numpy as np

class AudioWriter:
    def __init__(self):
        pass
    def add_sample(self,samples):
        if not hasattr(self,'data') :
            self.data = samples
        else:
            # print(self.data.shape)
            # print(samples.shape)
            # print('x')
            self.data= np.concatenate([self.data,samples])
    def write(self,fileName,frequency):
        mx=self.data.max()
        self.data/=mx
        write(fileName, frequency, self.data)
        