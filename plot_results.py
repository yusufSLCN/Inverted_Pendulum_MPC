import pickle
import os
import numpy as np
from closed_loop_cost import plot_results

if __name__ == "__main__":
    init_theta = np.pi/3
    exp_name = f'results_closedLoopCost_{init_theta:.2f}.pickle'
    exp_path = os.path.join('./results', exp_name)
    with open(exp_path, 'rb') as f:
        results = pickle.load(f)
    plot_results(results, f'Simulation Results Closed Loop Cost, Angle {init_theta:.2f}, Noise std 0.02')