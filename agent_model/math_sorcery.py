__author__ = 'richard'

def calculate_heading(self, vector_x, vector_y):  # TODO: export to trajectory
    theta = atan2(vector_y, vector_x)

    return theta*180/np.pi

def calculate_curvature(self): #TODO check if np arrays
    v_x, v_y, v_z = self.ensemble.velocity_x, self.ensemble.velocity_y, self.ensemble.velocity_z
    velo_vector = np.array([v_x, v_y, v_z]).T
    a_x, a_y, a_z = self.ensemble.acceleration_x, self.ensemble.acceleration_y, self.ensemble.acceleration_z
    accel_vector = np.array([a_x, a_y, a_z]).T
    # using formula from https://en.wikipedia.org/wiki/Curvature#Local_expressions_2
    for i in range(len(v_x)):
        k_i = np.abs( np.cross(velo_vector, accel_vector) ) /                           \
            (np.linalg.norm([a_x[i], a_y[i], a_z[i]]) ** 3)
    $ TODO: output?