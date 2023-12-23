from scipy.io.wavfile import read
import numpy as np

#wav to arr as j Samples,Channels
def wavToArr(filename):
    file = read(filename)
    arr = np.array(file[1])
    return arr
    
