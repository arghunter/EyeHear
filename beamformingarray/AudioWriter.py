from scipy.io.wavfile import write
import numpy as np

class AudioWriter:
    def __init__(self):
        pass
    def add_sample(self,samples,overlap):
        if not hasattr(self,'data') :
            self.data = samples
            self.dpos=overlap
            print(self.data.shape)
        else:
            # print(self.data.shape)
            # print(samples.shape)
            # print('x')
            print(self.data.shape)
            self.data=np.concatenate([self.data,np.zeros(((int(overlap)),self.data.shape[1]))])
            # self.data= np.concatenate([self.data,samples])
            self.data[self.dpos:len(self.data), :]+=samples
            self.dpos+=overlap
    def write(self,fileName,frequency):
        # mx=self.data.max()
        
        # if mx>1:
        # self.data/=mx
        write(fileName, frequency, self.data)
        