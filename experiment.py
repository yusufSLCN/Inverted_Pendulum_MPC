import numpy as np
import cv2
import pickle
import time
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from controlled_cart_and_pendulum import objective
from inverted_pendulum_model import InvertedPendulum
from inverted_pendulum_viz import InvertedPendulumViz
from casadi_.casadi_experiment import casadi_experiment


def experiment(solver_type, init_state, goal_state, args):
    sim_iter = 0
    state_logs = []
    error_logs = []
    time_logs = []
        
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
    #     bounds.append((-50, 50))
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
                 'vir_model':vir_model}

    # MPC Control Loop
    for i in range(num_steps):
        sim_iter = i

        # Run the optimizer
        st = time.time()
        result = minimize(objective, initial_guess, args=(args_dict),
                          method=solver_type, 
                          options={'disp': False})
        # print("Time taken for optimization: ", time.time() - st)
        # Extract optimal control inputs
        solver_time = time.time() - st
        print(solver_time)
        time_logs.append(solver_time)

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
            canvas = viz.step([pendulum_system.state.x, pendulum_system.state.v, pendulum_system.state.theta, pendulum_system.state.theta_dot], t=i * dt)
            # Display the canvas using cv2
            cv2.imshow('Inverted Pendulum', canvas)
        
        print(f'Sim-iter {i + 1} / {num_steps}: x= {init_state.x:.2f} / {goal_x:.2f}, v={init_state.v:.2f}, \
theta={init_state.theta:.2f} / {goal_theta:.2f}, input= {pendulum_system.inputs.force:.2f}')

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(int(dt*1000)) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    results = (state_logs, error_logs, time_logs, goal_x, goal_theta, sim_iter)

    return results

def plot_results(results, title):
    fig, axs = plt.subplots(3, 1, figsize=(10, 6))

    pos_title =  'Position and Angle - ' + title
    fig.suptitle(pos_title)
    times = []
    f = open(f"./results/results_{title}.txt", "w")
    
    for solver_type in results:
        state_logs, error_logs, time_logs, goal_x, goal_theta, sim_iter = results[solver_type]
        
        cart_poss = [s.x for s in state_logs]
        pendulum_angles = [s.theta for s in state_logs]
        idx = np.arange(len(cart_poss))

        # Plot cart position
        axs[0].plot(idx, cart_poss, label=solver_type)
        axs[0].set_xlabel('Iteration')
        axs[0].set_ylabel('Cart Position')

        # Plot pendulum angle
        axs[1].plot(idx, pendulum_angles, label=solver_type)
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Pendulum Angle')

        # Plot objective
        axs[2].plot(idx, error_logs, label=solver_type)
        axs[2].set_xlabel('Iteration')
        axs[2].set_ylabel('Error')

        times.append(time_logs)
        time_msg = f'{solver_type} --- Mean: {np.mean(time_logs):.2f}, Max: {np.max(time_logs):.2f}, Min: {np.min(time_logs):.2f} \n'
        f.write(time_msg)
        print(time_msg)

    f.close()

    axs[0].axhline(y=goal_x, color='red', linestyle='--', label='Target', linewidth=2)
    axs[1].axhline(y=goal_theta, color='red', linestyle='--', label='Target', linewidth=2)

    axs[0].legend(loc='upper right')
    axs[1].legend(loc='upper right')
    axs[2].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('./plots/' + pos_title + '.png')

    # Plot time
    fig = plt.figure()
    time_title = 'Time Measurements - ' + title
    plt.suptitle(time_title)
    plt.boxplot(times, labels=results.keys(), showfliers = False, whis = (0, 100), showmeans= True,
                    meanline= True, notch = False, showbox = False)
    # axs[3].violinplot(times, showmeans=True)
    plt.ylabel('Optimizer Time (s)')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()

    # Show the plots
    # plt.show()
    plt.savefig('./plots/' + time_title + '.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--render', action='store_true')
    parser.add_argument('-p', '--plot', action='store_true')
    args = parser.parse_args()
    exp_angles = [np.pi/3, np.pi, 3*np.pi/2]
    # exp_angles = [np.pi/3]
    optimization_methods = ['ipopt', 'SLSQP', 'BFGS', 'CG', 'Powell']
    # optimization_methods = ['SLSQP']

    for start_theta in exp_angles:
        exp_st = time.time()
        init_state = {'theta': start_theta, 'x':0}
        goal_state = {'theta': np.pi/2, 'x':1}
        results = {}
        for solver in optimization_methods:
            print(f'{solver=}')
            if solver == 'ipopt':
                result = casadi_experiment(solver, init_state, goal_state, args)
            else:
                result = experiment(solver, init_state, goal_state, args)
            results[solver] = result
        
        init_theta = init_state['theta']
        exp_time = time.time() - exp_st
        print(f'Experiment duration: {exp_time:.2f} s')
        # plot_results(results, f'Simulation Results wo Initial Guess Update Angle {init_theta:.2f}')
        exp_name = f'./results/results_{init_theta:.2f}.pickle'
        with open(exp_name, 'wb') as f:
            pickle.dump(results, f)
        plot_results(results, f'Simulation Results, Angle {init_theta:.2f}, Noise std 0.02')

        