from scipy.io.wavfile import read
import numpy as np
from queue import Queue
import sounddevice as sd
from AudioWriter import AudioWriter
#wav to arr as j Samples,Channels
class IOStream: #sample duration in microseconds
    def __init__(self,sample_duration=10000):
        self.sample_duration=sample_duration
        self.q=Queue()
        pass
    # Call one of the two below
    def wavToStream(self,filename):
        file = read(filename)
        self.arr = np.array(file[1])
        self.channels=len(file[1][0])
        self.frequency=file[0]
        dur=(1/self.frequency)*10**6
        n_samples = int(self.sample_duration/dur)
        iter=0
        
        while iter < self.arr.shape[0]:
            self.q.put(self.arr[iter:iter+n_samples])
            iter+=n_samples
        self.q.put(self.arr[iter: self.arr.shape[0]])
            
    def audio_callback(self,indata, frames, time, status):
        for i in range(frames):
            if(self.buffer_head<self.buffer.shape[0]):
                self.buffer[self.buffer_head]=indata[i]
                self.buffer_head+=1
            else:
                self.q.put(np.copy(self.buffer))
                self.buffer_head=0
        
            # print(frames)
        
    def streamAudio(self,frequency,channels):
        self.frequency=frequency
        self.channels=channels
        dur=(1/self.frequency)*10**6
        n_samples = int(self.sample_duration/dur)
        self.buffer=np.zeros((n_samples,channels))
        self.buffer_head=0
        stream=sd.InputStream(
            device=None, channels=self.channels,
            samplerate=self.frequency, callback=self.audio_callback)
        stream.start()
             
    def getNextSample(self):
        
        return self.q.get()

sd.default.device=18
io= IOStream()
io.streamAudio(48000,16)
# io.wavToStream("./beamformingarray/output.wav")
writer= AudioWriter()
for i in range(400):
    # print(type(io.getNextSample()))
    writer.add_sample(io.getNextSample())
# print(io.arr.shape)
# print(writer.data.shape)
writer.write("./beamformingarray/output2.wav",48000)   
