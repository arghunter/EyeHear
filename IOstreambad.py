# import sounddevice as sd
# import numpy as np
# import queue
# import sys
# from scipy.io.wavfile import write

# class IOStream():
#     def __init__(self,channels):
#         self.q = queue.Queue()
#         self.channels=[]
#         for i in range(channels):
#             self.channels.append(i)
#         sd.default.device=18
#         self.samplerate = 44100
#     def audio_callback(self,indata, frames, time, status):
        
#         if status:
#             print(status, file=sys.stderr)
#         # print(frames)
#         self.q.put(indata[::1, self.channels])
#     def start(self,loopFunc):
        
#         with sd.InputStream(
#             device=None, channels=len(self.channels),
#             samplerate=self.samplerate, callback=self.audio_callback):
#                 count=500
#                 while True:
#                     loopFunc(self,count)
#                     count-=1
#                     print(count)

# sound=np.zeros((2048,6))
# def loopFunc(io,count):
#     arr=io.q.get()
#     np.append(sound,arr)
   
#     if count==0:
#         write('output.wav', io.samplerate, sound)
#     # print(str(arr[0][0])+" "+str(arr[0][1])+" "+str(arr[0][2])+" "+str(arr[0][3])+" "+str(arr[0][4])+" "+str(arr[0][5]))
#     # print(str(np.mean(arr)))

# io=IOStream(6)
# io.start(loopFunc)
# # while True:
# #         print(io.q.get())