import numpy as np
import cv2
import time
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from controlled_cart_and_pendulum import objective
from inverted_pendulum_model import InvertedPendulum
from inverted_pendulum_viz import InvertedPendulumViz


def apply_wind_disturbance(self, wind_force):
    # Apply wind disturbance to the system
    # Modify the state based on the effects of the wind force

    # Extract state variables
    x1 = self.state.theta


    # Constants
    g = 9.81

    # Update state based on the effects of the wind force
    self.state.v += (-(self.M + self.m) * g * np.sin(x1) + wind_force) / (self.M + self.m)

disturbances_events = {'Wind': [], 'Mass': []}
def experiment2(solver_type, args):
    sim_iter = 0
    state_logs = []
    error_logs = []

    # Set simulation parameters
    dt = 0.05
    total_time = 10

    num_steps = int(total_time / dt)

    #  MPC Parameters
    P = 40  # Prediction horizon

    # Goal angle
    goal_theta = np.pi / 2.0
    goal_x = 1.0

    # Cost function weights
    eth_W = 100.0
    ex_W = 100.0
    f_rate_W = 0.01

    # Bounds
    bounds = []
    for _ in range(P):
        bounds.append((-100, 100))

    # Initialize the model and MPC optimizer
    pendulum_system = InvertedPendulum(m=0.5, M=5.0, L=1)
    # pendulum_system.uncertainty_gaussian_std = 0.02

    # Instantiate the model and visualization classes
    viz = InvertedPendulumViz(x_start=-5, x_end=5, pendulum_len=1)

    # Initial state
    pendulum_system.state = pendulum_system.State(cart_position=0.0, pendulum_angle=1)
    init_state = pendulum_system.state

    # Initial guess for the control inputs
    initial_guess = np.zeros(P)

    # define args_dict
    args_dict = {'goal_theta': goal_theta,
                 'goal_x': goal_x,
                 'init_state': init_state,
                 'P': P,
                 'eth_W': eth_W, 'ex_W': ex_W, 'f_rate_W': f_rate_W,
                 'dt': dt,
                 'm': pendulum_system.m, 'M': pendulum_system.M, 'L': pendulum_system.L}
    # MPC Control Loop
    original_mass = pendulum_system.m
    disturbance_type = None

    for i in range(num_steps):
        sim_iter = i
        args_dict['init_state'] = init_state
        # Run the optimizer
        st = time.time()



        # Run the simulation with the perturbed mass and observe the impact on performance metrics

        result = minimize(objective, initial_guess, args=(args_dict),
                          method=solver_type,
                          options={'disp': False})

        if i == 5:
            wind_force = 0.2
            apply_wind_disturbance(pendulum_system, wind_force)
            disturbance_type = "Wind"
            disturbances_events['Wind'].append(('Wind', i, 0, i, 1))  # Tuple: (Type, Start time, Position on plot)
        if i == 10:
            # Uncertainty modeling: random perturbation to mass
            uncertainty_factor = np.random.uniform(-0.05, 0.05)  # Example uncertainty range
            perturbed_mass = original_mass * (1 + uncertainty_factor)
            pendulum_system.m = perturbed_mass
            disturbance_type = "Mass"
            disturbances_events['Mass'].append(('Mass', i, 0, i, 1))  # Tuple: (Type, Start time, Position on plot)

        state_logs.append(init_state)
        error_logs.append(result.fun)

        # print("Time taken for optimization: ", time.time() - st)
        # Extract optimal control inputs
        optimal_controls = result.x

        # Apply the first control input to the system
        pendulum_system.inputs.force = optimal_controls[0]
        pendulum_system.step(dt)

        # Update the initial state
        init_state = pendulum_system.state

        if np.abs(init_state.x - goal_x) / goal_x < 0.05 and np.abs(init_state.theta - goal_theta) / goal_theta < 0.05:
            break

        # Update the initial guess
        next_init_guess = np.zeros_like(initial_guess)
        next_init_guess[:-1] = optimal_controls[1:]
        initial_guess = next_init_guess

        # Restore the original mass for the next iteration
        pendulum_system.m = original_mass

        # Visualize the current state using the visualization class
        if args.render:
            canvas = viz.step([pendulum_system.state.x, pendulum_system.state.v, pendulum_system.state.theta,
                               pendulum_system.state.theta_dot], t=i * dt)
            # Display the canvas using cv2
            cv2.imshow('Inverted Pendulum', canvas)

        print(f'Sim-iter {i + 1} / {num_steps}: x= {init_state.x:.2f} / {goal_x:.2f}, v={init_state.v:.2f}, \
    theta={init_state.theta:.2f} / {goal_theta:.2f}, input= {pendulum_system.inputs.force:.2f}')

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(int(dt * 1000)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    if args.plot:
        cart_poss = [s.x for s in state_logs]
        pendulum_angles = [s.theta for s in state_logs]
        idx = np.arange(len(cart_poss))

        fig, axs = plt.subplots(3, 1, figsize=(10, 6))

        title = f'{solver_type} Simulation Results'
        fig.suptitle(title)

        # Plot cart position
        axs[0].plot(idx, cart_poss, label='Cart Position')
        axs[0].axhline(y=goal_x, color='red', linestyle='--', label='Target', linewidth=2)
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Position')
        axs[0].legend()

        # Plot pendulum angle
        axs[1].plot(idx, pendulum_angles, label='Pendulum Angle')
        axs[1].axhline(y=goal_theta, color='red', linestyle='--', label='Target', linewidth=2)
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Angle')
        axs[1].legend()

        # Objective
        axs[2].plot(idx, error_logs, label='Error')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Error')
        axs[2].legend()

        # Adjust layout
        plt.tight_layout()

        # Show the plots
        plt.show()
        # plt.savefig(title + '.png')

    return (state_logs, error_logs, goal_x, goal_theta, sim_iter)


def plot_results(results, title):
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle(title)

    for solver_type in results:
        state_logs, error_logs, goal_x, goal_theta, sim_iter = results[solver_type]

        cart_poss = [s.x for s in state_logs]
        pendulum_angles = [s.theta for s in state_logs]
        idx = np.arange(len(cart_poss))
        # Plot cart position
        axs[0].plot(idx, cart_poss, label=solver_type)
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Cart Position after perturbed mass')

        # Plot pendulum angle
        axs[1].plot(idx, pendulum_angles, label=solver_type)
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Pendulum Angle after perturbed mass')

        # Objective
        axs[2].plot(idx, error_logs, label=solver_type)
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Error after perturbed mass')
        # Annotate disturbance events if provided
        for event_type, event_list in disturbances_events.items():
            for disturbance_event in event_list:
                if disturbance_event[1] < len(cart_poss):
                    if disturbance_event[0] == 'Wind':
                        c = 'red'
                        marker = 'o'
                    else:
                        c = 'blue'
                        marker = 'x'   # Use 'o' for 'Wind' and 'x' for other type
                    axs[0].plot(disturbance_event[1], cart_poss[disturbance_event[1]], c='red', marker=marker)
                    axs[1].plot(disturbance_event[1], pendulum_angles[disturbance_event[1]], c='red', marker=marker)
                    axs[2].plot(disturbance_event[1], error_logs[disturbance_event[1]], c='red', marker=marker)

    axs[0].axhline(y=goal_x, color='red', linestyle='--', label='Target', linewidth=2)
    axs[1].axhline(y=goal_theta, color='red', linestyle='--', label='Target', linewidth=2)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    # plt.show()
    plt.savefig(title + '.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--render', action='store_true')
    parser.add_argument('-p', '--plot', action='store_true')
    args = parser.parse_args()

    optimization_methods = ['SLSQP', 'BFGS', 'L-BFGS-B']

    results = {}
    for solver in optimization_methods:
        result = experiment2(solver, args)
        results[solver] = result
    plot_results(results, 'Simulation Results after perturbed mass and wind only once')
