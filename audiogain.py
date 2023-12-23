import sounddevice as sd
import numpy as np  # Make sure NumPy is loaded before it is used in the callback
  # avoid "imported but unused" message (W0611)
import queue
import sys
q = queue.Queue()
channels=[1]
mapping = [c - 1 for c in channels]
print(mapping)
def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::1, mapping])
    
device_info = sd.query_devices(None, 'input')
samplerate = device_info['default_samplerate']
with sd.InputStream(
        device=None, channels=max([1]),
        samplerate=samplerate, callback=audio_callback):
    while True:
        print(str(np.mean(q.get())*1))
    
  