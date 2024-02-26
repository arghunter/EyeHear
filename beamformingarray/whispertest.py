import whisper
import numpy as np
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import scipy.signal as signal
from IOStream import IOStream
from AudioWriter import AudioWriter
from VAD import VAD
from MUSIC import MUSIC
import threading
import pickle
from DelayApproximation import DelayAproximator
model = whisper.load_model("tiny.en")

# result=model.transcribe("./beamformingarray/AudioTests/test_input_sig.wav")

# print(result["text"])

io=IOStream()
aw=AudioWriter()
file = read("./beamformingarray/AudioTests/test_input_sig.wav")
pcm=np.array(file[1])/32767
print(pcm.shape)
io.arrToStream(pcm,48000)
# while(not io.complete()):
#     sample=io.getNextSample()
#     # print(sample)
#     aw.add_sample(sample,480)
data=pcm.T[0].astype(np.float32).reshape((-1,1))
print(data.shape)
# audio = whisper.pad_or_trim(data)
mel = whisper.log_mel_spectrogram(data).to(model.device)
options = whisper.DecodingOptions(fp16=False)
result = whisper.decode(model, mel, options)
print(result.text)
aw.write("./beamformingarray/AudioTests/10.wav",48000)