# Make something where oyu pass in source coords and the coord of the mics and then i calculates the time diff from each mic from the first mic
import numpy as np
v=343.3
class DelayAproximator:

    def __init__(self,coords) :
        self.coords=coords

    def get_delays(self,pos):
        distances = []
        for mic_pos in self.coords:
            dx = mic_pos[0] - pos[0]
            dy = mic_pos[1] - pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            distances.append(distance)
        reference_distance = distances[0]
        tdoa_values = []
        for distance in distances:
            time_diff = (distance - reference_distance) / v
            tdoa_values.append(time_diff)
        return tdoa_values
    


approx=DelayAproximator([[0,0],[0.028,0],[0.056,0],[0.084,0],[0.112,0],[0.14,0],[0.168,0],[0.196,0]])
pos=[-1,10]
delays=approx.get_delays(pos)
print(delays)