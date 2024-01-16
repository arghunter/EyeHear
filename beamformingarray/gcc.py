import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
N = 8000; 

Fs = 48000; 

Ts = 1/Fs; #

Ndelay = 100;
rsig = np.random.randn(N*2, 1)
 

x = rsig[0:N];  
print(x.shape)
y=rsig[Ndelay:(Ndelay + N)]
 
X = np.fft.fft(x,axis=0);

Y = np.fft.fft(y,axis=0);

R = np.multiply(X,np.conj(Y));

tphat = np.real(np.fft.ifft(np.divide(R,np.abs(R)),axis=0));
tphat=np.reshape(tphat,(-1))
locs, _ = signal.find_peaks(tphat, height=None, distance=None)
sorted_indices = np.argsort(tphat[locs])[::-1]
pks = tphat[locs][sorted_indices]
locs = locs[sorted_indices]
print(pks)