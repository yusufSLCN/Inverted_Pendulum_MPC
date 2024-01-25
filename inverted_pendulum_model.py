import numpy as np

class InvertedPendulum:
    def __init__(self, L=1.0, m=0.1, M=1.0, g=9.8):
        self.L = L  # Length of pendulum
        self.m = m  # Mass of the bob
        self.M = M  # Mass of the cart
        self.g = g  # Gravitational acceleration
        self.uncertainty_gaussian_std = 0.0

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

    def step_euler(self, dt=0.01):
        final_state = self._step(self.state, dt)
        # Update the internal state
        self.state = final_state

        return final_state

    def add(self, state1, state2, factor):
        result = self.State()
        result.x = state1.x + state2.x * factor
        result.v = state1.v + state2.v * factor
        result.theta = (state1.theta + state2.theta * factor) % (2 * np.pi)
        result.theta_dot = state1.theta_dot + state2.theta_dot * factor
        return result

    def step_rk4(self, dt=0.01):
        k1 = self._step(self.state, dt)

        k2_start_state = self.add(self.state, k1, 0.5)
        k2 = self._step(k2_start_state, dt * 1.5)

        k3_start_state = self.add(self.state, k2, 0.5)
        k3 = self._step(k3_start_state, dt * 1.5)

        k4_start_state = self.add(self.state, k3, 1)
        k4 = self._step(k4_start_state, dt)

        sum_state = self.add(k1, k2, 2)
        sum_state = self.add(sum_state, k3, 2)
        sum_state = self.add(sum_state, k4, 1)
        final_state = self.add(self.state, sum_state, dt / 6)

        # Update the internal state
        self.state = final_state
        return final_state
    
    
    def _step(self, cur_state, dt):
        final_state = self.State()

        # Unpack state variables
        x, v, theta, theta_dot = cur_state.x, cur_state.v, cur_state.theta, cur_state.theta_dot

        # Runge-Kutta integration
        k1x = v * dt
        k1v = (self.inputs.force - self.m * self.L * theta_dot**2 * np.cos(theta) + self.m * self.g * np.cos(theta) * np.sin(theta)) / (self.M + self.m - self.m * np.sin(theta)**2) * dt
        k1theta = theta_dot * dt
        k1theta_dot = (-self.g/self.L * np.cos(theta) - 1./self.L * np.sin(theta) * (self.inputs.force - self.m * self.L * theta_dot**2 * np.cos(theta) + self.m * self.g * np.cos(theta) * np.sin(theta)) / (self.M + self.m - self.m * np.sin(theta)**2)) * dt

        k2x = (v + 0.5 * k1v) * dt
        k2v = (self.inputs.force - self.m * self.L * (theta_dot + 0.5 * k1theta_dot)**2 * np.cos(theta + 0.5 * k1theta) + self.m * self.g * np.cos(theta + 0.5 * k1theta) * np.sin(theta + 0.5 * k1theta)) / (self.M + self.m - self.m * np.sin(theta + 0.5 * k1theta)**2) * dt
        k2theta = (theta_dot + 0.5 * k1theta_dot) * dt
        k2theta_dot = (-self.g/self.L * np.cos(theta + 0.5 * k1theta) - 1./self.L * np.sin(theta + 0.5 * k1theta) * (self.inputs.force - self.m * self.L * (theta_dot + 0.5 * k1theta_dot)**2 * np.cos(theta + 0.5 * k1theta) + self.m * self.g * np.cos(theta + 0.5 * k1theta) * np.sin(theta + 0.5 * k1theta)) / (self.M + self.m - self.m * np.sin(theta + 0.5 * k1theta)**2)) * dt

        k3x = (v + 0.5 * k2v) * dt
        k3v = (self.inputs.force - self.m * self.L * (theta_dot + 0.5 * k2theta_dot)**2 * np.cos(theta + 0.5 * k2theta) + self.m * self.g * np.cos(theta + 0.5 * k2theta) * np.sin(theta + 0.5 * k2theta)) / (self.M + self.m - self.m * np.sin(theta + 0.5 * k2theta)**2) * dt
        k3theta = (theta_dot + 0.5 * k2theta_dot) * dt
        k3theta_dot = (-self.g/self.L * np.cos(theta + 0.5 * k2theta) - 1./self.L * np.sin(theta + 0.5 * k2theta) * (self.inputs.force - self.m * self.L * (theta_dot + 0.5 * k2theta_dot)**2 * np.cos(theta + 0.5 * k2theta) + self.m * self.g * np.cos(theta + 0.5 * k2theta) * np.sin(theta + 0.5 * k2theta)) / (self.M + self.m - self.m * np.sin(theta + 0.5 * k2theta)**2)) * dt

        k4x = (v + k3v) * dt
        k4v = (self.inputs.force - self.m * self.L * (theta_dot + k3theta_dot)**2 * np.cos(theta + k3theta) + self.m * self.g * np.cos(theta + k3theta) * np.sin(theta + k3theta)) / (self.M + self.m - self.m * np.sin(theta + k3theta)**2) * dt
        k4theta = (theta_dot + k3theta_dot) * dt
        k4theta_dot = (-self.g/self.L * np.cos(theta + k3theta) - 1./self.L * np.sin(theta + k3theta) * (self.inputs.force - self.m * self.L * (theta_dot + k3theta_dot)**2 * np.cos(theta + k3theta) + self.m * self.g * np.cos(theta + k3theta) * np.sin(theta + k3theta)) / (self.M + self.m - self.m * np.sin(theta + k3theta)**2)) * dt

        damping_theta = -0.5 * theta_dot
        damping_x = -1.0 * v

        final_state.x = x + dt*(k1x + 2*k2x + 2*k3x + k4x) / 6.0
        final_state.v = v + dt*(k1v + 2*k2v + 2*k3v + k4v)/ 6.0 + damping_x * dt
        final_state.theta = theta + dt*(k1theta + 2*k2theta + 2*k3theta + k4theta) / 6.0
        final_state.theta_dot = theta_dot + dt*(k1theta_dot + 2*k2theta_dot + 2*k3theta_dot + k4theta_dot) / 6.0 + damping_theta*dt

        if self.uncertainty_gaussian_std > 0:
            # to simulate uncertainty in the model and sensors
            final_state.theta += np.random.normal(0, self.uncertainty_gaussian_std)

        # Update the internal state
        self.state = final_state

        return final_state
    
    # TODO: Add Linearized model and Linearized model step function
    
    def apply_disturbance(self, angle_disturbance, pos_disturbance=0):
        self.state.theta += angle_disturbance
        self.state.x += pos_disturbance
    
    def set_uncertainty_gaussian_std(self, std):
        self.uncertainty_gaussian_std = std
    
    def reset(self):
        self.state = self.State()
        self.inputs = self.Inputs()
        

