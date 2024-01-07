import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class MVDR:
    def __init__(self,vector_gen, sample_rate=48000):
        self.vector_gen = vector_gen
        self.sample_rate=sample_rate
    def get_peaks(self, frame):
        thetas= np.arange(-180,180,1)
        res=[]
        for theta in thetas:
            look=self.vector_gen.get_look_vector(theta)
            cov=frame@frame.H # Covariance
            cov_inv=np.linalg.pinv(cov)
            cost=1/(look.H@cov_inv@look)
            cost=np.abs(cost[0,0])
            cost=10*np.log10(cost)
            res.append(cost)
        res/=np.max(res)
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(np.radians(thetas), res) # MAKE SURE TO USE RADIAN FOR POLAR
        ax.set_theta_zero_location('N') # make 0 degrees point up
        ax.set_theta_direction(-1) # increase clockwise
        ax.set_rlabel_position(30)  # Move grid labels away from other labels
        plt.show()



class Look_Vector_Generator:
    #r_pos: array of microphone positions
    def __init__(self,r_pos):
        self.r_pos=r_pos
        
    def get_look_vector(self,theta):
        return (np.asmatrix(np.exp(-2j*np.pi*self.r_pos*np.sin(np.radians(theta))))).T
d=0.15
r_pos=np.arange(4)*d
look_gen=Look_Vector_Generator(r_pos)
mvdr=MVDR(look_gen)
sample_rate = 48000
N = 10000 # number of samples to simulate

# Create a tone to act as the transmitted signal
t = np.arange(N)/sample_rate
Nr = 8 # 8 elements
theta1 = -40 / 180 * np.pi # convert to radians
theta2 = 30 / 180 * np.pi
theta3 = -40 / 180 * np.pi
a1 = np.asmatrix(np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1)))
a2 = np.asmatrix(np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2)))
a3 = np.asmatrix(np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3)))
# we'll use 3 different frequencies
r = a1.T @ np.asmatrix(np.exp(2j*np.pi*500*t)) + \
    a2.T @ np.asmatrix(np.exp(2j*np.pi*1000*t)) + \
    0 * a3.T @ np.asmatrix(np.exp(2j*np.pi*2000*t))
n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
r = r + 0.04*n
mvdr.get_peaks(r)