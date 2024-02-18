import numpy as np
import cv2
import time
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
        next_state = vir_model.step_rk4(dt)
        # Penalize distance from goal angle
        Error += eth_W * np.abs(next_state.theta - goal_theta) ** 2
        # Penalize distance from goal position
        Error += ex_W * np.abs(next_state.x - goal_x) ** 2
        # Penalize control effort
        # Error += x[i] ** 2
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

    # Bounds
    bounds = []
    for _ in range(P):
        bounds.append((-50, 50))

    # Initialize the model and MPC optimizer
    pendulum_system = InvertedPendulum(m=.1, M=5.0, L=0.3)
    # pendulum_system.uncertainty_gaussian_std = 0.02
    
    # Instantiate the model and visualization classes    
    viz = InvertedPendulumViz(x_start=-5, x_end=5, pendulum_len=1)

    # Create an instance of the InteractivePlot class
    interactive_plot = InteractivePlot()
    
    # Initial state
    pendulum_system.state = pendulum_system.State(cart_position=0.0, pendulum_angle=1.57)
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


def run_mpc_simulation(dt, total_time, prediction_horizon, goal_theta, goal_x, eth_W, ex_W, f_rate_W, bounds, solver):
    # Set simulation parameters
    num_steps = int(total_time / dt)
    
    # Initialize the model and MPC optimizer
    pendulum_system = InvertedPendulum(m=.1, M=5.0, L=0.3)
    
    # Initial state
    pendulum_system.state = pendulum_system.State(cart_position=0.0, pendulum_angle=1.57)
    init_state = pendulum_system.state
        
    # Initial guess for the control inputs
    initial_guess = np.zeros(prediction_horizon)
    
    # define args_dict
    args_dict = {'goal_theta': goal_theta,
                 'goal_x': goal_x, 
                 'init_state': init_state, 
                 'P': prediction_horizon, 
                 'eth_W': eth_W, 'ex_W': ex_W, 'f_rate_W': f_rate_W, 
                 'dt': dt,
                 'm': pendulum_system.m, 'M': pendulum_system.M, 'L': pendulum_system.L}

    inputs_list = []
    loss_values = []
    opti_time = []
    angle_list = []
    pos_list = []
    # MPC Control Loop
    for i in range(num_steps):

        # Run the optimizer
        st = time.time()
        if solver == 'SLSQP':
            result = minimize(objective, initial_guess, args=(args_dict),
                              method='SLSQP', bounds=bounds, 
                              options={'disp': True})
        elif solver == 'L-BFGS-B':
            result = minimize(objective, initial_guess, args=(args_dict),
                              method='L-BFGS-B',
                              options={'disp': True})
        elif solver == 'CG':
            result = minimize(objective, initial_guess, args=(args_dict),
                              method='CG', 
                              options={'disp': True})
        elif solver == 'Powell':
            result = minimize(objective, initial_guess, args=(args_dict),
                                method='Powell', bounds=bounds, 
                                options={'disp': True})
            
        opti_time.append(time.time() - st)
        # Extract optimal control inputs
        optimal_controls = result.x
        
        # clip the control inputs if they are out of bounds just for CG and L-BFGS-B
        if solver in ['CG', 'L-BFGS-B']:
            optimal_controls = np.clip(optimal_controls, bounds[0][0], bounds[0][1])

        # Apply the first control input to the system
        pendulum_system.inputs.force = optimal_controls[0]        
        pendulum_system.step_rk4(dt)
        
        # Update the initial state
        init_state = pendulum_system.state
        args_dict['init_state'] = init_state
        
        inputs_list.append(optimal_controls[0])
        
        angle_list.append(pendulum_system.state.theta)
        pos_list.append(pendulum_system.state.x)
        
        # Calculate loss
        loss = objective(optimal_controls, args_dict)
        loss_values.append(loss)
        
    return loss_values, opti_time, inputs_list, angle_list, pos_list


def log_values(log_file_path, loss_values):
    with open(log_file_path, 'w') as f:
        for loss in loss_values:
            f.write(f"{loss}\n")
    print(f"Loss values logged to {log_file_path}")
    
def closed_loop_cost(loss_values, dt):
    return np.sum(loss_values) * dt

if __name__=="__main__":
    # Test different solvers
    solvers = ['SLSQP', 'Powell']
    
    for solver in solvers:
        
        # Test different prediction horizons (P)
        for prediction_horizon in [10, 20]:
            print(f"Testing solver: {solver}, Prediction Horizon: {prediction_horizon}")
            
            # Set file path for logging loss values
            loss_log_file_path = f"loss_values_{solver.lower()}_{prediction_horizon}.txt"
            input_log_file_path = f"input_values_{solver.lower()}_{prediction_horizon}.txt"
            angle_log_file_path = f"angle_values_{solver.lower()}_{prediction_horizon}.txt"
            pos_log_file_path = f"pos_values_{solver.lower()}_{prediction_horizon}.txt"
            time_log_file_path = f"time_values_{solver.lower()}_{prediction_horizon}.txt"
            
            
                
            # Set simulation parameters
            dt = 0.05
            total_time = 25.0
            num_steps = int(total_time / dt)
            
            # Goal and weights
            goal_theta = np.pi / 2.0
            goal_x = -1.0
            eth_W = 100.0
            ex_W = 100.0
            f_rate_W = 0.1

            # Bounds
            bounds = []
            for _ in range(prediction_horizon):
                bounds.append((-50, 50))

            # Run MPC simulation
            loss_values, opti_time, inputs_list, angle_list, pos_list = run_mpc_simulation(dt, total_time, prediction_horizon, goal_theta, goal_x, eth_W, ex_W, f_rate_W, bounds, solver)

            # Log loss values to a file
            log_values(loss_log_file_path, loss_values)
            log_values(input_log_file_path, inputs_list)
            log_values(angle_log_file_path, angle_list)
            log_values(pos_log_file_path, pos_list)
            log_values(time_log_file_path, opti_time)
            
            # Calculate closed loop cost and write to file
            closed_loop_cost_val = closed_loop_cost(loss_values, dt)
            with open(f"closed_loop_cost_{solver.lower()}_{prediction_horizon}.txt", 'w') as f:
                f.write(f"{closed_loop_cost_val}\n")
                # min and mean and max of optimization time
                f.write(f"Optimization time: {np.mean(opti_time)}\n")
                f.write(f"Optimization time std: {np.std(opti_time)}\n")
                f.write(f"Optimization time max: {np.max(opti_time)}\n")
                # min and mean and max of loss values
                f.write(f"Min loss: {np.min(loss_values)}\n")
                f.write(f"Mean loss: {np.mean(loss_values)}\n")
                f.write(f"Max loss: {np.max(loss_values)}\n")
                # min and mean and max of inputs
                f.write(f"Min input: {np.min(inputs_list)}\n")
                f.write(f"Mean input: {np.mean(inputs_list)}\n")
                f.write(f"Max input: {np.max(inputs_list)}\n")
                
            