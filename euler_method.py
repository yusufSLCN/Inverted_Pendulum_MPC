def step(self, dt=0.01):
    final_state = self.State()

    # Unpack state variables
    x, v, theta, theta_dot = self.state.x, self.state.v, self.state.theta, self.state.theta_dot

    # Update state variables
    x_ddot = self.inputs.force - self.m * self.L * theta_dot * theta_dot * np.cos(theta) + self.m * self.g * np.cos(
        theta) * np.sin(theta)
    x_ddot = x_ddot / (self.M + self.m - self.m * np.sin(theta) * np.sin(theta))

    theta_ddot = -self.g / self.L * np.cos(theta) - 1. / self.L * np.sin(theta) * x_ddot

    damping_theta = -0.5 * theta_dot
    damping_x = -1.0 * v

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