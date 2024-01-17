import numpy as np
import cv2
import time
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from controlled_cart_and_pendulum import objective
from inverted_pendulum_model import InvertedPendulum
from inverted_pendulum_viz import InvertedPendulumViz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--render', action='store_true')
    parser.add_argument('-p', '--plot', action='store_true')
    args = parser.parse_args()

    if args.plot:
        state_logs = []
        error_logs = []

    # Set simulation parameters
    dt = 0.01
    total_time = 2.0
    num_steps = int(total_time / dt)
    
    #  MPC Parameters
    P = 80  # Prediction horizon

    # Goal angle
    goal_theta = np.pi / 2.0
    goal_x = 1.0

    # Cost function weights
    eth_W = 100.0
    ex_W = 100.0
    f_rate_W = 0.1

    # Bounds
    bounds = []
    for _ in range(P):
        bounds.append((-500, 500))

    # Initialize the model and MPC optimizer
    pendulum_system = InvertedPendulum(m=0.1, M=5.0, L=1)
    # pendulum_system.uncertainty_gaussian_std = 0.02
    
    # Instantiate the model and visualization classes    
    viz = InvertedPendulumViz(x_start=-5, x_end=5, pendulum_len=1)
    
    # Initial state
    init_angle = np.pi / 2.5
    pendulum_system.state = pendulum_system.State(cart_position=0.0, pendulum_angle=init_angle)
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
    for i in range(num_steps):
        args_dict['init_state'] = init_state
        # Run the optimizer
        st = time.time()
        result = minimize(objective, initial_guess, args=(args_dict),
                          method='SLSQP', bounds=bounds, 
                          options={'disp': True})
        
        if args.plot:
            state_logs.append(init_state)
            error_logs.append(result.fun)

        # print("Time taken for optimization: ", time.time() - st)
        # Extract optimal control inputs
        optimal_controls = result.x

        # Apply the first control input to the system
        pendulum_system.inputs.force = optimal_controls[0]        
        # pendulum_system.step_euler(dt)
        pendulum_system.step_rk4(dt)
        
        # Update the initial state
        init_state = pendulum_system.state
        if np.abs(init_state.x - goal_x)/goal_x < 0.05 and np.abs(init_state.theta - goal_theta)/ goal_theta < 0.05:
            break
        
        #Update the initial guess
        next_init_guess = np.zeros_like(initial_guess)
        next_init_guess[:-1] = optimal_controls[1:]
        initial_guess = next_init_guess

        # at time = 50, apply disturbance
        # if i == 500:
        #     pendulum_system.apply_disturbance(0.1)

        # Visualize the current state using the visualization class
        if args.render:
            canvas = viz.step([pendulum_system.state.x, pendulum_system.state.v, pendulum_system.state.theta, pendulum_system.state.theta_dot], t=i * dt)
            # Display the canvas using cv2
            cv2.imshow('Inverted Pendulum', canvas)
        
        print(f'Sim-iter {i + 1} / {num_steps}: x= {init_state.x:.2f} / {goal_x:.2f}, v={init_state.v:.2f}, theta={init_state.theta:.2f} / {goal_theta:.2f}')

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    if args.plot:
        cart_poss = [s.x for s in state_logs]
        pendulum_angles = [s.theta for s in state_logs]
        idx = np.arange(len(cart_poss))

        fig, axs = plt.subplots(3, 1, figsize=(10, 6))

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

        