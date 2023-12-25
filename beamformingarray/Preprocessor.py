import numpy as np


class Preprocessor:
    # Mirroered: channels 1 and 2 are iodentical, 3 and 4 are identical and so on. Culls half the channels
    def __init__(self,mirrored=True):
        self.mirrored = mirrored
    def process(self,samples):
        if self.mirrored:
            arr=np.zeros((samples.shape[0],int(samples.shape[1]/2)))
            for i in range(samples.shape[0]):
                
                for j in range(len(arr[0])):
                    arr[i][j]=samples[i,j*2]
               
            return arr
        else:
            return samples
        
        
