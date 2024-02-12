import threading
import numpy as np
from BeamformerMVDR import Beamformer
from queue import Queue
from IOStream import IOStream
from AudioWriter import AudioWriter
class MVDRasync:
    
    def __init__(self,ioStream,sample_rate=48000,spacing=np.array([0,0.028,0.056,0.084,0.112,0.14,0.168,0.196]),num_channels=8,exp_avg=50,frame_len=960,stft_len=1024):
        self.spacing=spacing
        self.num_channels=num_channels
        self.exp_avg=exp_avg
        self.frame_len=frame_len
        self.stft_len=stft_len
        self.sample_rate=sample_rate
        self.mvdr=Beamformer(sample_rate,spacing,num_channels,exp_avg,frame_len,stft_len)
        self.q=Queue()
        self.io=ioStream
        self.t= threading.Thread(target=self.start_beamforming)
        self.t.start()
    def start_beamforming(self):
        print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
        self.io.getNextSample()
        self.io.getNextSample()
        self.io.getNextSample()
        self.io.getNextSample()
        self.io.getNextSample()
        while(True):
            frame=self.io.getNextSample()
            print(frame.shape)
            data=self.mvdr.beamform(frame)
            print(data.shape)
            self.q.put(frame)
        
import sounddevice as sd
sd.default.device=18
stream = IOStream(20000,10000)
stream.streamAudio(48000,8)      
mvdr=MVDRasync(stream)
aw=AudioWriter()
for i in range(0,10):
    aw.add_sample((mvdr.q.get()),480)
    
aw.write("./AudioTests/async1.wav",48000)
