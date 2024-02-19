import pickle

import numpy as np
import cv2
import time
import argparse
import matplotlib.pyplot as plt
from casadi_.closed_loop_cost import casadi_experiment
from scipy.optimize import minimize
from controlled_cart_and_pendulum import objective
from inverted_pendulum_model import InvertedPendulum
from inverted_pendulum_viz import InvertedPendulumViz


closed_loop_cost_value = 0.00
closed_loop_cost_values = []
def close_loop_cost(solver_type,init_state, goal_state, args):
    sim_iter = 0
    state_logs = []
    error_logs = []

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

    # Bounds
    # bounds = []
    # for _ in range(P):
    #     bounds.append((-100, 100))

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

    global closed_loop_cost_value
    global closed_loop_cost_values
    closed_loop_cost_value = 0.00
    closed_loop_cost_values = []

    # MPC Control Loop
    for i in range(num_steps):
        sim_iter = i
        # Run the optimizer
        st = time.time()

        # Run the simulation with the perturbed mass and observe the impact on performance metrics
        result = minimize(objective, initial_guess, args=(args_dict),
                          method=solver_type,
                          options={'disp': False})
        closed_loop_cost_values.append(closed_loop_cost_value)
        closed_loop_cost_value += dt * result.fun

        print(closed_loop_cost_value)
        # print("Time taken for optimization: ", time.time() - st)
        # Extract optimal control inputs
        optimal_controls = result.x

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
        error_logs.append(result.fun)

        # Update the initial guess
        next_init_guess = np.zeros_like(initial_guess)
        next_init_guess[:-1] = optimal_controls[1:]
        initial_guess = next_init_guess

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

        axs[3].plot(idx, closed_loop_cost_values, label='Cost')
        axs[3].set_xlabel('Time')
        axs[3].set_ylabel('Closed Loop Cost')
        axs[3].legend()

        # Adjust layout
        plt.tight_layout()

        # Show the plots
        plt.show()
        # plt.savefig(title + '.png')

    return (state_logs, error_logs, closed_loop_cost_values, goal_x, goal_theta, sim_iter)

def plot_results(results, title):
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle(title)
    f = open(f"Closed loop cost_progress_{title}.txt", "w")

    for i, solver_type in enumerate(results):
        state_logs, error_logs, closed_loop_cost_value_prog, goal_x, goal_theta, sim_iter = results[solver_type]

        cart_poss = [s.x for s in state_logs]
        pendulum_angles = [s.theta for s in state_logs]
        idx = np.arange(len(cart_poss))
        # # Plot cart position
        # axs[0].plot(idx, cart_poss, label=solver_type)
        # axs[0].set_xlabel('Time')
        # axs[0].set_ylabel('Cart Position')
        #
        # # Plot pendulum angle
        # axs[1].plot(idx, pendulum_angles, label=solver_type)
        # axs[1].set_xlabel('Time')
        # axs[1].set_ylabel('Pendulum Angle')
        #
        # # Objective
        # axs[2].plot(idx, error_logs, label=solver_type)
        # axs[2].set_xlabel('Time')
        # axs[2].set_ylabel('Error')


        colors = {'ipopt': 'magenta', 'SLSQP': 'blue', 'BFGS': 'orange', 'CG': 'green', 'Powell': 'red' }
        y_values = np.arange(len(colors))
        ax.plot(closed_loop_cost_value_prog, label=solver_type, color=colors[solver_type])
        ax.set_xlabel('Time')
        ax.set_ylabel('Closed Loop Cost')
        ax.legend()

        time_msg = f'{solver_type} --- Closed loop cost prog: {str(closed_loop_cost_value_prog)} \n'
        f.write(time_msg)
        print(time_msg)

    f.close()

    # axs[0].axhline(y=goal_x, color='red', linestyle='--', label='Target', linewidth=2)
    # axs[1].axhline(y=goal_theta, color='red', linestyle='--', label='Target', linewidth=2)
    #
    # axs[0].legend()
    # axs[1].legend()
    # axs[2].legend()
    ax.legend()

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

    optimization_methods = ['ipopt', 'SLSQP', 'BFGS', 'CG', 'Powell']
    init_state = {'theta': np.pi/3, 'x': 0}
    goal_state = {'theta': np.pi / 2, 'x': 1}
    results = {}
    for solver in optimization_methods:
        print(f'{solver=}')
        if solver == 'ipopt':
            result = casadi_experiment(solver, init_state, goal_state, args)
        else:
            result = close_loop_cost(solver, init_state, goal_state, args)
        results[solver] = result

    init_theta = init_state['theta']
    exp_name = f'./results/results_closedLoopCostProg_{init_theta:.2f}.pickle'
    with open(exp_name, 'wb') as f:
        pickle.dump(results, f)
    plot_results(results, f'Simulation Results for closed loop cost Progress {init_theta:.2f}')
