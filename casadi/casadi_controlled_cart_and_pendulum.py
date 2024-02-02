import numpy as np
import cv2
import time
import casadi as ca
import sys
sys.path.append('..')
from inverted_pendulum_viz import InvertedPendulumViz
from inverted_pendulum_model import InvertedPendulum





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
        #  Penalize control changes
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

    # Init optimizer 
    opti = ca.Opti()
    U = opti.variable(P)
    opti.solver('ipopt')

    # Constraints
    # for i in range(P):
    #     opti.subject_to(U[i] < 10)
    #     opti.subject_to(U[i] > -10)


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

        loss = objective(U, args_dict)
        opti.minimize(loss)

        # Solve the optimization problem
        opti.set_initial(U, initial_guess)


        sol = opti.solve()

        print(time.time() - st)

        # Apply the first control input to the system
        optimal_controls = sol.value(U)
        pendulum_system.inputs.force = optimal_controls[0]        
        pendulum_system.step_rk4(dt)
        
        # Update the initial state
        init_state = pendulum_system.state
        args_dict['init_state'] = init_state
        
        inputs_list.append(optimal_controls[0])

        # Update the initial guess
        initial_guess[:-1] = optimal_controls[1:]
        initial_guess[-1] = 0
        # print(initial_guess)

        
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