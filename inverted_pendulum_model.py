import numpy as np
import os

class InvertedPendulum:
    def __init__(self, L=1.0, m=0.1, M=1.0, g=9.8, uncertainty_gaussian_std = 0):
        self.L = L  # Length of pendulum
        self.m = m  # Mass of the bob
        self.M = M  # Mass of the cart
        self.g = g  # Gravitational acceleration
        self.uncertainty_gaussian_std = uncertainty_gaussian_std

        if uncertainty_gaussian_std > 0:
            self.noise = np.random.normal(0, uncertainty_gaussian_std, 2000)
            
        self.sim_iter = 0

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

    def step_rk4(self, dt=0.01):
        
        final_state = self.State()

        # Unpack state variables
        x, v, theta, theta_dot = self.state.x, self.state.v, self.state.theta, self.state.theta_dot

        # Update state variables
        k1_x_dot = v
        k1_v_dot = self.inputs.force - self.m * self.L* theta_dot * theta_dot * np.cos(theta) + self.m * self.g * np.cos(theta) * np.sin(theta)
        k1_v_dot = k1_v_dot / (self.M + self.m - self.m * np.sin(theta) * np.sin(theta))
        k1_theta_dot = theta_dot
        k1_theta_ddot = -self.g/self.L * np.cos(theta) - 1./self.L * np.sin(theta) * k1_v_dot

        k2_x_dot = v + k1_v_dot * dt/2
        k2_v_dot = self.inputs.force - self.m * self.L* k1_theta_dot * k1_theta_dot * np.cos(theta + k1_theta_dot * dt/2) + self.m * self.g * np.cos(theta + k1_theta_dot * dt/2) *  np.sin(theta + k1_theta_dot * dt/2)
        k2_v_dot = k2_v_dot / (self.M + self.m - self.m * np.sin(theta + k1_theta_dot * dt/2) * np.sin(theta + k1_theta_dot * dt/2))
        k2_theta_dot = theta_dot + k1_theta_ddot * dt/2
        k2_theta_ddot = -self.g/self.L * np.cos(theta + k1_theta_dot * dt/2) - 1./self.L * np.sin(theta + k1_theta_dot * dt/2) * k2_v_dot

        k3_x_dot = v + k2_v_dot * dt / 2
        k3_v_dot = self.inputs.force - self.m * self.L * k2_theta_dot * k2_theta_dot * np.cos(
            theta + k2_theta_dot * dt / 2) + self.m * self.g * np.cos(theta + k2_theta_dot * dt / 2) * np.sin(
            theta + k2_theta_dot * dt / 2)
        k3_v_dot = k3_v_dot / (self.M + self.m - self.m * np.sin(theta + k2_theta_dot * dt / 2) * np.sin(
            theta + k2_theta_dot * dt / 2))
        k3_theta_dot = theta_dot + k2_theta_ddot * dt / 2
        k3_theta_ddot = -self.g / self.L * np.cos(theta + k2_theta_dot * dt / 2) - 1. / self.L * np.sin(
            theta + k2_theta_dot * dt / 2) * k3_v_dot

        k4_x_dot = v + k3_v_dot * dt
        k4_v_dot = self.inputs.force - self.m * self.L * k3_theta_dot * k3_theta_dot * np.cos(
            theta + k3_theta_dot * dt) + self.m * self.g * np.cos(theta + k3_theta_dot * dt) * np.sin(
            theta + k3_theta_dot * dt)
        k4_v_dot = k4_v_dot / (
                    self.M + self.m - self.m * np.sin(theta + k3_theta_dot * dt) * np.sin(theta + k3_theta_dot * dt))
        k4_theta_dot = theta_dot + k3_theta_ddot * dt
        k4_theta_ddot = -self.g / self.L * np.cos(theta + k3_theta_dot * dt) - 1. / self.L * np.sin(
            theta + k3_theta_dot * dt) * k4_v_dot
        
        damping_theta = -0.5 * theta_dot
        damping_x =  -1.0 * v

        # Update state variables
        final_state.x = x + dt / 6 * (k1_x_dot + 2 * k2_x_dot + 2 * k3_x_dot + k4_x_dot)
        final_state.v = v + dt / 6 * (k1_v_dot + 2 * k2_v_dot + 2 * k3_v_dot + k4_v_dot) + damping_x * dt
        final_state.theta = theta + dt / 6 * (k1_theta_dot + 2 * k2_theta_dot + 2 * k3_theta_dot + k4_theta_dot)
        final_state.theta_dot = theta_dot + dt / 6 * (
                    k1_theta_ddot + 2 * k2_theta_ddot + 2 * k3_theta_ddot + k4_theta_ddot) + damping_theta * dt
        # add noise
        if self.uncertainty_gaussian_std > 0:
            # to simulate uncertainty in the model and sensors
            if self.sim_iter >= len(self.noise):
                self.sim_iter = 0

            final_state.theta += self.noise[self.sim_iter]
            
        # Update the internal state
        self.state = final_state
        
        self.sim_iter += 1
        return final_state
    
    
    def step_euler(self, dt=0.01):
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

    def apply_wind_disturbance(self, dt=0.01, wind_force=0.0):
        final_state = self.State()

        # Unpack state variables
        x, v, theta, theta_dot = self.state.x, self.state.v, self.state.theta, self.state.theta_dot

        # Runge-Kutta integration
        k1x = v * dt
        k1v = (self.inputs.force - self.m * self.L * theta_dot ** 2 * np.cos(theta) + self.m * self.g * np.cos(
            theta) * np.sin(theta)
               + wind_force * np.sin(theta)) / (self.M + self.m - self.m * np.sin(theta) ** 2) * dt
        k1theta = theta_dot * dt
        k1theta_dot = (-self.g / self.L * np.cos(theta) - 1. / self.L * np.sin(theta) *
                       (self.inputs.force - self.m * self.L * theta_dot ** 2 * np.cos(theta) + self.m * self.g * np.cos(
                           theta) * np.sin(theta)
                        + wind_force * np.sin(theta)) / (self.M + self.m - self.m * np.sin(theta) ** 2)) * dt

        k2x = (v + 0.5 * k1v) * dt
        k2v = (self.inputs.force - self.m * self.L * (theta_dot + 0.5 * k1theta_dot) ** 2 * np.cos(
            theta + 0.5 * k1theta) + self.m * self.g * np.cos(theta + 0.5 * k1theta) * np.sin(theta + 0.5 * k1theta)
               + wind_force * np.sin(theta + 0.5 * k1theta)) / (
                          self.M + self.m - self.m * np.sin(theta + 0.5 * k1theta) ** 2) * dt
        k2theta = (theta_dot + 0.5 * k1theta_dot) * dt
        k2theta_dot = (-self.g / self.L * np.cos(theta + 0.5 * k1theta) - 1. / self.L * np.sin(theta + 0.5 * k1theta) *
                       (self.inputs.force - self.m * self.L * (theta_dot + 0.5 * k1theta_dot) ** 2 * np.cos(
                           theta + 0.5 * k1theta) + self.m * self.g * np.cos(theta + 0.5 * k1theta) * np.sin(
                           theta + 0.5 * k1theta)
                        + wind_force * np.sin(theta + 0.5 * k1theta)) / (
                                   self.M + self.m - self.m * np.sin(theta + 0.5 * k1theta) ** 2)) * dt

        k3x = (v + 0.5 * k2v) * dt
        k3v = (self.inputs.force - self.m * self.L * (theta_dot + 0.5 * k2theta_dot) ** 2 * np.cos(
            theta + 0.5 * k2theta) + self.m * self.g * np.cos(theta + 0.5 * k2theta) * np.sin(theta + 0.5 * k2theta)
               + wind_force * np.sin(theta + 0.5 * k2theta)) / (
                          self.M + self.m - self.m * np.sin(theta + 0.5 * k2theta) ** 2) * dt
        k3theta = (theta_dot + 0.5 * k2theta_dot) * dt
        k3theta_dot = (-self.g / self.L * np.cos(theta + 0.5 * k2theta) - 1. / self.L * np.sin(theta + 0.5 * k2theta) *
                       (self.inputs.force - self.m * self.L * (theta_dot + 0.5 * k2theta_dot) ** 2 * np.cos(
                           theta + 0.5 * k2theta) + self.m * self.g * np.cos(theta + 0.5 * k2theta) * np.sin(
                           theta + 0.5 * k2theta)
                        + wind_force * np.sin(theta + 0.5 * k2theta)) / (
                                   self.M + self.m - self.m * np.sin(theta + 0.5 * k2theta) ** 2)) * dt

        k4x = (v + k3v) * dt
        k4v = (self.inputs.force - self.m * self.L * (theta_dot + k3theta_dot) ** 2 * np.cos(
            theta + k3theta) + self.m * self.g * np.cos(theta + k3theta) * np.sin(theta + k3theta)
               + wind_force * np.sin(theta + k3theta)) / (self.M + self.m - self.m * np.sin(theta + k3theta) ** 2) * dt
        k4theta = (theta_dot + k3theta_dot) * dt
        k4theta_dot = (-self.g / self.L * np.cos(theta + k3theta) - 1. / self.L * np.sin(theta + k3theta) *
                       (self.inputs.force - self.m * self.L * (theta_dot + k3theta_dot) ** 2 * np.cos(
                           theta + k3theta) + self.m * self.g * np.cos(theta + k3theta) * np.sin(theta + k3theta)
                        + wind_force * np.sin(theta + k3theta)) / (
                                   self.M + self.m - self.m * np.sin(theta + k3theta) ** 2)) * dt

        damping_theta = -0.5 * theta_dot
        damping_x = -1.0 * v

        final_state.x = x + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
        final_state.v = v + dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0 + damping_x * dt
        final_state.theta = theta + dt * (k1theta + 2 * k2theta + 2 * k3theta + k4theta) / 6.0
        final_state.theta_dot = theta_dot + dt * (
                    k1theta_dot + 2 * k2theta_dot + 2 * k3theta_dot + k4theta_dot) / 6.0 + damping_theta * dt

        if self.uncertainty_gaussian_std > 0:
            # to simulate uncertainty in the model and sensors
            final_state.theta += np.random.normal(0, self.uncertainty_gaussian_std)

        # Update the internal state
        self.state = final_state

        return final_state

    def set_uncertainty_gaussian_std(self, std):
        self.uncertainty_gaussian_std = std
    
    def reset(self):
        self.state = self.State()
        self.inputs = self.Inputs()
        
