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
    def get_pos(angle,dist):
        angle=np.radians(angle)
        pos=[dist*np.cos(angle),dist*np.sin(angle)]
        print(pos)
        return pos

    


