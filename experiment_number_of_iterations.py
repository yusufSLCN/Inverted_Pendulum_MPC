import pickle

import numpy as np
import cv2
import time
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from casadi_.number_of_it import casadi_experiment
from controlled_cart_and_pendulum import objective
from inverted_pendulum_model import InvertedPendulum
from inverted_pendulum_viz import InvertedPendulumViz


def number_of_iterations(solver_type,init_state, goal_state, args):
    sim_iter = 0
    state_logs = []
    error_logs = []
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
        # Run the optimizer
        st = time.time()

        # Run the simulation with the perturbed mass and observe the impact on performance metrics
        result = minimize(objective, initial_guess, args=(args_dict),
                          method=solver_type,
                          options={'disp': False})
        convergence_rate_values.append(result.nit)

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

    return (state_logs, error_logs, convergence_rate_values, goal_x, goal_theta, sim_iter)

def plot_results(results, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle(title)
    convergence_rates =[]
    f = open(f"number_of_it_{title}.txt", "w")

    for i, solver_type in enumerate(results):
        state_logs, error_logs, convergence_rate_values, goal_x, goal_theta, sim_iter = results[solver_type]

        cart_poss = [s.x for s in state_logs]
        pendulum_angles = [s.theta for s in state_logs]
        idx = np.arange(len(cart_poss))


        convergence_rates.append(convergence_rate_values)
        time_msg = f'{solver_type} --- Mean: {int(np.mean(convergence_rate_values))}, Max: {np.max(convergence_rate_values):.2f}, Min: {np.min(convergence_rate_values):.2f} \n'
        f.write(time_msg)
        print(time_msg)

    f.close()

    # Convergence rate
    ax.boxplot(convergence_rates, labels=results.keys(), showfliers=False, whis=(0, 100),showmeans=True,
                    meanline=True, notch=False, showbox=False)
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of iterations')
    plt.yscale('log')

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
            result = number_of_iterations(solver, init_state, goal_state, args)
        results[solver] = result

    init_theta = init_state['theta']
    exp_name = f'./results/results_numberOfIterations_{init_theta:.2f}.pickle'
    with open(exp_name, 'wb') as f:
        pickle.dump(results, f)
    plot_results(results, f'Simulation Results for Number of iterations Angle {init_theta:.2f}')
