import threading
import numpy as np
from beamformerMVDR import Beamformer
from queue import Queue
from IOStream import IOStream
from AudioWriter import AudioWriter
class MVDRasync:
    
    def __init__(self,ioStream,sample_rate=48000,spacing=np.array([[-0.08,0.042],[-0.08,0.014],[-0.08,-0.028],[-0.08,-0.042],[0.08,0.042],[0.08,0.014],[0.08,-0.028],[0.08,-0.042]]),num_channels=8,exp_avg=50,frame_len=960,stft_len=1024):
        self.spacing=spacing
        self.num_channels=num_channels
        self.exp_avg=exp_avg
        self.frame_len=frame_len
        self.stft_len=stft_len
        self.sample_rate=sample_rate
        self.mvdr=Beamformer(sample_rate,spacing,num_channels,exp_avg,frame_len,stft_len)
        self.q=Queue()
        self.dq=Queue()
        self.io=ioStream
        self.mvdr.set_doa(90)
        self.t= threading.Thread(target=self.start_beamforming)
        self.t.start()
        
    def start_beamforming(self):
        print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")

        while(True):
            # print(" ")
            frame=self.io.get()
            self.dq.put(frame)
            # print(frame.shape)
            data=self.mvdr.beamform(20*frame*32767)
            # print(data.shape)
            self.q.put(data)
    def beamform(self, sample):
        data=self.mvdr.beamform(sample)
        return data
        
# import sounddevice as sd
# sd.default.device=18
# stream = IOStream(20000,10000)
# stream.streamAudio(48000,8)      
# aw=AudioWriter()
# stream.getNextSample()
# stream.getNextSample()
# stream.getNextSample()
# stream.getNextSample()
# mvdr=MVDRasync(stream)
# for i in range(0,250):
#     print(i)
#     aw.add_sample(mvdr.q.get(),480)
#     # aw.add_sample(mvdr.beamform(100*stream.getNextSample()),160)
#     # aw.add_sample(stream.getNextSample(),160)

# print()
# aw.write("./beamformingarray/AudioTests/0async1.wav",48000)
