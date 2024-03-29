import numpy as np
import cv2
import time
import argparse
import matplotlib.pyplot as plt
import casadi as ca
import sys

sys.path.append('..')
from inverted_pendulum_viz import InvertedPendulumViz
from inverted_pendulum_model import InvertedPendulum
from controlled_cart_and_pendulum import objective


def casadi_experiment(solver_type, init_state, goal_state, args):
    sim_iter = 0
    state_logs = []
    error_logs = []
    time_logs = []
    convergence_rate_values = []

    # Set simulation parameters
    dt = 0.05
    total_time = 8.0
    num_steps = int(total_time / dt)

    #  MPC Parameters
    P = 20  # Prediction horizon

    # Init state
    init_x = init_state['x']
    init_theta = init_state['theta']

    # Goal state
    goal_theta = goal_state['theta']
    goal_x = goal_state['x']

    # Cost function weights
    eth_W = 100.0
    ex_W = 100.0
    f_rate_W = 0.01

    # Init optimizer
    opti = ca.Opti()
    U = opti.variable(P)
    opti.solver(solver_type)

    # Constraints
    # for i in range(P):
    #     opti.subject_to(U[i] <= 80)
    #     opti.subject_to(U[i] >= -80)

    clip_value = 80

    # Initialize the model and MPC optimizer
    pendulum_system = InvertedPendulum(m=0.1, M=5.0, L=0.3, uncertainty_gaussian_std=0.02)

    # pendulum_system.uncertainty_gaussian_std = 0.02

    # Instantiate the model and visualization classes
    viz = InvertedPendulumViz(x_start=-5, x_end=5, pendulum_len=1)

    # Initial state
    pendulum_system.state = pendulum_system.State(cart_position=init_x, pendulum_angle=init_theta)
    init_state = pendulum_system.state

    # Initial guess for the control inputs
    initial_guess = np.zeros(P)

    # Virtual model
    vir_model = InvertedPendulum(m=pendulum_system.m, M=pendulum_system.M, L=pendulum_system.L)

    # define args_dict
    args_dict = {'goal_theta': goal_theta,
                 'goal_x': goal_x,
                 'init_state': init_state,
                 'P': P,
                 'eth_W': eth_W, 'ex_W': ex_W, 'f_rate_W': f_rate_W,
                 'dt': dt,
                 'm': pendulum_system.m, 'M': pendulum_system.M, 'L': pendulum_system.L,
                 'vir_model': vir_model}

    # MPC Control Loop
    for i in range(num_steps):
        sim_iter = i
        st = time.time()
        # Run the optimizer
        loss = objective(U, args_dict)
        opti.minimize(loss)

        # Solve the optimization problem
        opti.set_initial(U, initial_guess)

        st = time.time()

        sol = opti.solve()

        convergence_rate_values.append(sol.stats()['iter_count'])

        # Apply the first control input to the system
        optimal_controls = sol.value(U)
        pendulum_system.inputs.force = optimal_controls[0]

        # Apply the first control input to the system
        clipped_force = clip_value if optimal_controls[0] >= clip_value else optimal_controls[0]
        clipped_force = -clip_value if optimal_controls[0] <= -clip_value else clipped_force
        pendulum_system.inputs.force = clipped_force

        # pendulum_system.inputs.force = optimal_controls[0]
        pendulum_system.step_rk4(dt)

        # Update the initial state
        init_state = pendulum_system.state
        args_dict['init_state'] = init_state

        state_logs.append(init_state)
        error_logs.append(sol.value(opti.f))

        # Update the initial guess
        initial_guess[:-1] = optimal_controls[1:]
        initial_guess[-1] = 0

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

    return (state_logs, error_logs, convergence_rate_values, goal_x, goal_theta, sim_iter)


def plot_results(results, title):
    fig, axs = plt.subplots(4, 1, figsize=(10, 6))
    fig.suptitle(title)
    conv_rates = []
    f = open(f"results_{title}.txt", "w")

    for solver_type in results:
        state_logs, error_logs, convergence_rate_values, goal_x, goal_theta, sim_iter = results[solver_type]

        cart_poss = [s.x for s in state_logs]
        pendulum_angles = [s.theta for s in state_logs]
        idx = np.arange(len(cart_poss))

        # Plot cart position
        axs[0].plot(idx, cart_poss, label=solver_type)
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Cart Position')

        # Plot pendulum angle
        axs[1].plot(idx, pendulum_angles, label=solver_type)
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Pendulum Angle')

        # Objective
        axs[2].plot(idx, error_logs, label=solver_type)
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Error')

        conv_rates.append(convergence_rate_values)
        conv_rate_msg = f'{solver_type} --- Mean: {np.mean(convergence_rate_values):.2f}, Max: {np.max(convergence_rate_values):.2f}, Min: {np.min(convergence_rate_values):.2f} \n'
        f.write(conv_rate_msg)
        print(conv_rate_msg)

    f.close()

    # Time
    axs[3].boxplot(conv_rates, labels=results.keys(), showfliers=False, whis=(0, 100))
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('Convergence rate')
    plt.yscale('log')

    axs[0].axhline(y=goal_x, color='red', linestyle='--', label='Target', linewidth=2)
    axs[1].axhline(y=goal_theta, color='red', linestyle='--', label='Target', linewidth=2)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()

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

    optimization_methods = ['ipopt']
    # optimization_methods = ['SLSQP']
    init_state = {'theta': np.pi/3, 'x': 0}
    goal_state = {'theta': np.pi / 2, 'x': 1}
    results = {}
    for solver in optimization_methods:
        result = casadi_experiment(solver, init_state, goal_state, args)
        results[solver] = result

    init_theta = init_state['theta']
    plot_results(results, f'Closed loop Simulation Results Angle {init_theta:.2f}')