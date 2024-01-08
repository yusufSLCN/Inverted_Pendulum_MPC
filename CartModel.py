import numpy as np

class CartModel:
    def __init__(self, L, m, M, g=9.8):
        self.g = g # Gravitational Acceleration
        self.L = L # Length of pendulum

        self.m = m #mass of bob (kg)
        self.M = M  # mass of cart (kg)

    def step(self, t, y):
        x_ddot = self.u(t) - self.m * self.L*y[3] * y[3] * np.cos( y[2] ) + self.m * self.g*np.cos(y[2]) *  np.sin(y[2])
        x_ddot = x_ddot / (self.M+self.m-self.m* np.sin(y[2]) * np.sin(y[2]))

        theta_ddot = -self.g/self.L * np.cos( y[2] ) - 1./self.L * np.sin( y[2] ) * x_ddot

        damping_theta =  -0.5 * y[3]
        damping_x =  -1.0 * y[1]

        return [ y[1], x_ddot + damping_x, y[3], theta_ddot + damping_theta]
    
    def u(self, t):
        t1 = 3
        t2 = 3.5

        if( t > t1 and t < t2 ):
            return 50
        else:
            return 0