import pickle
import os
import numpy as np
from number_of_iterations import plot_results

if __name__ == "__main__":
    init_theta = np.pi/3
    exp_name = f'results_convergenceRate_{init_theta:.2f}.pickle'
    exp_path = os.path.join('./results', exp_name)
    with open(exp_path, 'rb') as f:
        results = pickle.load(f)
    plot_results(results, f'Simulation Results Number of Iterations, Angle {init_theta:.2f}, Noise std 0.02')