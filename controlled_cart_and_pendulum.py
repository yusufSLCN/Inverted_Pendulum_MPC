import numpy as np
import cv2
import time
from scipy.optimize import minimize
from inverted_pendulum_model import InvertedPendulum
from inverted_pendulum_viz import InvertedPendulumViz


# MPC Objective Function
def objective(x, args_dict):
    
    # Unpack the arguments
    goal_theta = args_dict['goal_theta']
    goal_x = args_dict['goal_x']
    init_state = args_dict['init_state']
    P = args_dict['P']
    eth_W = args_dict['eth_W']
    ex_W = args_dict['ex_W']
    dt = args_dict['dt']
    vir_model = args_dict['vir_model']
    
    Error = 0
    init_state_1 = init_state
    # Initialize the virtual model
    vir_model.state = init_state_1
    for i in range(P):
        vir_model.inputs.force = x[i]
        next_state = vir_model.step_rk4(dt)
        # Penalize distance from goal angle
        Error += eth_W * (next_state.theta - goal_theta) ** 2
        # Penalize distance from goal position
        Error += ex_W * (next_state.x - goal_x) ** 2
        # Penalize control effort
        Error += 0.1 * (x[i] ** 2)

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
    total_time = 100.0
    num_steps = int(total_time / dt)
    
    #  MPC Parameters
    P = 20  # Prediction horizon

    # Goal angle
    goal_theta = np.pi / 2.0
    goal_x = -1.0

    # Cost function weights
    eth_W = 100.0
    ex_W = 100.0
    f_rate_W = 0.1

    # Bounds
    bounds = []
    for _ in range(P):
        bounds.append((-50, 50))

    # Initialize the model and MPC optimizer
    pendulum_system = InvertedPendulum(m=.1, M=5.0, L=0.3)
    # pendulum_system.uncertainty_gaussian_std = 0.02
    
    # Instantiate the model and visualization classes    
    viz = InvertedPendulumViz(x_start=-5, x_end=5, pendulum_len=1)

    # Initial state
    pendulum_system.state = pendulum_system.State(cart_position=0.0, pendulum_angle=1.57)
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

    inputs_list = []
    # MPC Control Loop
    for i in range(num_steps):

        # Run the optimizer
        st = time.time()
        result = minimize(objective, initial_guess, args=(args_dict),
                          method='SLSQP', bounds=bounds, 
                          options={'disp': True})
        # print("Time taken for optimization: ", time.time() - st)
        # Extract optimal control inputs
        optimal_controls = result.x
        print(time.time() - st)
        # Apply the first control input to the system
        pendulum_system.inputs.force = optimal_controls[0]        
        pendulum_system.step_rk4(dt)
        
        # Update the initial state
        init_state = pendulum_system.state
        args_dict['init_state'] = init_state
        
        inputs_list.append(optimal_controls[0])

        
        # at time = 50, apply disturbance
        if i == 75:
            pendulum_system.apply_disturbance(0.1)
        
        # Visualize the current state using the visualization class
        canvas = viz.step([pendulum_system.state.x, pendulum_system.state.v, pendulum_system.state.theta, pendulum_system.state.theta_dot], t=i * dt)

        # Display the canvas using cv2
        cv2.imshow('Inverted Pendulum', canvas)
                
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(int(dt*1000)) & 0xFF == ord('q'):
            break
        
        
    cv2.destroyAllWindows()




if __name__=="__main__":
    main()