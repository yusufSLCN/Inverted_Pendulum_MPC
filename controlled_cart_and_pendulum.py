import numpy as np
import cv2
import time

from matplotlib import pyplot as plt
from scipy.optimize import minimize
from inverted_pendulum_model import InvertedPendulum
from inverted_pendulum_viz import InvertedPendulumViz
from interactive_plot import InteractivePlot

# MPC Objective Function
def objective(x, args_dict):
    
    # Unpack the arguments
    goal_theta = args_dict['goal_theta']
    goal_x = args_dict['goal_x']
    init_state = args_dict['init_state']
    P = args_dict['P']
    eth_W = args_dict['eth_W']
    ex_W = args_dict['ex_W']
    f_rate_W = args_dict['f_rate_W']
    dt = args_dict['dt']
    m = args_dict['m']
    M = args_dict['M']
    L = args_dict['L']
    
    Error = 0
    init_state_1 = init_state
    # Initialize the virtual model
    vir_model = InvertedPendulum(m=m, M=M, L=L)
    vir_model.state = init_state_1
    for i in range(P):
        vir_model.inputs.force = x[i]
        next_state = vir_model.step(dt)
        # Penalize distance from goal angle
        Error += eth_W * np.abs(next_state.theta - goal_theta) ** 2
        # Penalize distance from goal position
        Error += ex_W * np.abs(next_state.x - goal_x) ** 2
        # Penalize control effort
        # Error += np.abs(x[i]) ** 2
        # # # Penalize control changes
        # if i > 0:
        #     Error += f_rate_W * np.abs(x[i] - x[i - 1]) ** 2

        init_state_1 = next_state
    return Error

def update_angle_goal(current_time):
    """
    Update the angle goal based on a sine wave.

    Parameters:
    - current_time: Current time.

    Returns:
    - angle_goal: Updated angle goal.
    """
    amplitude = 1
    frequency = 0.1
    offset = np.pi/2
    t = current_time
    angle_goal = amplitude * np.sin(2 * np.pi * frequency * t + offset)
    return angle_goal


def main():
    
    # Set simulation parameters
    dt = 0.05
    total_time = 10.0
    num_steps = int(total_time / dt)

    #  MPC Parameters
    P = 40  # Prediction horizon

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

    # List of optimization methods
    optimization_methods = ['SLSQP', 'BFGS', 'L-BFGS-B']

    # Dictionary to store convergence information foqqr each method
    convergence_info_by_method = {method: {'values': [], 'num_iterations': []} for method in optimization_methods}

    # MPC Control Loop
    # Run optimization for each method
    for method in optimization_methods:
        print(method)
        pendulum_system.state = pendulum_system.State(cart_position=0.0, pendulum_angle=1)
        init_state = pendulum_system.state
        # Initial guess for the control inputs
        initial_guess = np.zeros(P)
        for i in range(num_steps):
            args_dict['init_state'] = init_state
            result = minimize(objective, initial_guess, args=(args_dict), method=method,
                              options={'disp': False})
            optimal_controls = result.x
            # Store objective function value at this time step
            convergence_info_by_method[method]['values'].append(result.fun)
            print(result.nit)
            convergence_info_by_method[method]['num_iterations'].append(result.nit)  # Store number of iterations

            # Apply the first control input to the system (outside the inner loop)
            pendulum_system.inputs.force = optimal_controls[0]
            pendulum_system.step(dt)

            # Update the initial state
            init_state = pendulum_system.state
            args_dict['init_state'] = init_state

            # Update the initial guess
            next_init_guess = np.zeros_like(initial_guess)
            next_init_guess[:-1] = optimal_controls[1:]
            initial_guess = next_init_guess

            # at time = 50, apply disturbance
            # if i == 50:
            #     pendulum_system.apply_disturbance(0.1)

            # Visualize the current state using the visualization class
            canvas = viz.step([pendulum_system.state.x, pendulum_system.state.v, pendulum_system.state.theta,
                               pendulum_system.state.theta_dot], t=i * dt)

            # Display the canvas using cv2
            cv2.imshow('Inverted Pendulum', canvas)

            # # Break the loop if 'q' key is pressed
            if cv2.waitKey(int(dt*1000)) & 0xFF == ord('q'):
                break


    for method in optimization_methods:
        # convergence_info_by_method[method]['num_iterations'][:20] = array_of_methods[i]
        num_iterations = convergence_info_by_method[method]['num_iterations'][:30]
        print(num_iterations)
        plt.plot(num_iterations, label=method)

    plt.figure()
    plt.xlabel('Time Step')
    plt.ylabel('Number of Iterations')
    plt.title('Number of Iterations Comparison of Optimization Methods')
    plt.legend()
    plt.show()
    input("Press Enter to continue...")

    cv2.destroyAllWindows()



    # Plot number of iterations for each method





if __name__=="__main__":
    main()