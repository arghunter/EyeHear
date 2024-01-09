import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from scipy import fft
from Signal import Sine
from SignalGen import SignalGen
from IOStream import IOStream
from AudioWriter import AudioWriter

class MVDR:
    def __init__(self,vector_gen, sample_rate=48000):
        self.vector_gen = vector_gen
        self.sample_rate=sample_rate
    def get_peaks(self, frame):
        # frame=(self.convert_to_complex(frame))
        thetas= np.arange(-180,180,1)
        res=[]
        for theta in thetas:
            look=self.vector_gen.get_look_vector(theta)
            cov=frame @frame.H# Covariance
            cov_inv=np.linalg.pinv(cov)
            cost=1/(look.H@cov_inv@look)
            cost=np.abs(cost[0,0])
            cost = 10*np.log10(cost)
            res.append(cost)
        res/=np.max(res)
        peaks=thetas[(signal.find_peaks(res)[0])]
        print(peaks)
        # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # ax.plot(np.radians(thetas), res) # MAKE SURE TO USE RADIAN FOR POLAR
        # ax.set_theta_zero_location('N') # make 0 degrees point up
        # ax.set_theta_direction(-1) # increase clockwise
        # ax.set_rlabel_position(30)  # Move grid labels away from other labels
        # plt.show()
            # print(thetas[peaks[i]])
    def convert_to_complex(self,frame):
        arr=np.matrix(frame.astype(complex))
        # print(arr)
        # for i in range(len(arr)):
        #     print(arr[i])
        return arr




class Look_Vector_Generator:
    #r_pos: array of microphone positions
    def __init__(self,r_pos):
        self.r_pos=r_pos
        
    def get_look_vector(self,theta):
        return (np.asmatrix(np.exp(-2j*np.pi*self.r_pos*np.sin(np.radians(theta))))).T
d=0.10
Nr=8
r_pos=np.arange(Nr)*d
look_gen=Look_Vector_Generator(r_pos)
mvdr=MVDR(look_gen)

# 8 elements
N=14400
sample_rate=48000
# t = np.arange(N)/sample_rate
# theta1 = 90 / 180 * np.pi # convert to radians
# theta2 = 25 / 180 * np.pi
# theta3 = -40 / 180 * np.pi
# a1 = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1))
# print(a1)
# a1=np.ones(Nr)
# a1 = a1.reshape(-1,1)
sig=Sine(frequency=0.01e6)
tone1=sig.generate_wave(0.3)

gen=SignalGen(Nr,d)
gen.update_delays(0)
tone1=gen.delay_and_gain(tone1)
io=IOStream()
io.arrToStream(tone1,sample_rate=sample_rate)
while (not io.complete()):
    frame=io.getNextSample()
    spectrum=fft.fft(frame)
    mvdr.get_peaks((np.asmatrix(spectrum)).T)
    print(spectrum.shape)



# tone1=signal.hilbert(tone1)
# we'll use 3 different frequencies
# tone1 = np.exp(2j*np.pi*0.01e6*t)

# print(tone1)
r = tone1.reshape(Nr,-1)
# r=tone1
print(r.shape)
# for i in range(len(r.T)):
#     print(r.T[i])
# n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
# r = r + 0.04*n
# r=(gen.delay_and_gain(sig.generate_wave(0.1)))

# mvdr.get_peaks(np.asmatrix(r))