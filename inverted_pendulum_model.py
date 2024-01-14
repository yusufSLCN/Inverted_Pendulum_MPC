import numpy as np

class InvertedPendulum:
    def __init__(self, L=1.0, m=0.1, M=1.0, g=9.8):
        self.L = L  # Length of pendulum
        self.m = m  # Mass of the bob
        self.M = M  # Mass of the cart
        self.g = g  # Gravitational acceleration

        self.state = self.State()
        self.inputs = self.Inputs()

    class State:
        def __init__(self, cart_position=0, cart_velocity=0, pendulum_angle=0, pendulum_angular_velocity=0):
            self.x = cart_position
            self.v = cart_velocity
            self.theta = pendulum_angle
            self.theta_dot = pendulum_angular_velocity

    class Inputs:
        def __init__(self, force=0):
            self.force = force

    def step(self, dt=0.01):
        final_state = self.State()

        # Unpack state variables
        x, v, theta, theta_dot = self.state.x, self.state.v, self.state.theta, self.state.theta_dot

        # Update state variables
        x_ddot = self.inputs.force - self.m * self.L* theta_dot * theta_dot * np.cos(theta) + self.m * self.g * np.cos(theta) *  np.sin(theta)
        x_ddot = x_ddot / (self.M + self.m - self.m * np.sin(theta) * np.sin(theta))

        theta_ddot = -self.g/self.L * np.cos(theta) - 1./self.L * np.sin(theta) * x_ddot

        damping_theta =  -0.5 * theta_dot
        damping_x =  -1.0 * v

        # Update state variables
        final_state.x = x + v * dt
        final_state.v = v + x_ddot * dt + damping_x * dt
        final_state.theta = theta + theta_dot * dt
        final_state.theta_dot = theta_dot + theta_ddot * dt + damping_theta * dt

        # Update the internal state
        self.state = final_state

        return final_state
    
    # TODO: Add Linearized model and Linearized model step function
    
    def apply_disturbance(self, angle_disturbance, pos_disturbance=0):
        self.state.theta += angle_disturbance
        self.state.x += pos_disturbance
        

