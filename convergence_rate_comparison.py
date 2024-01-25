from matplotlib import pyplot as plt

array_of_methods = [[52, 57, 46, 44, 43, 48, 47, 49, 29, 33, 31, 36, 35, 34, 37, 32, 31, 27, 27, 30, 32, 15, 15, 30, 15, 15, 14, 16, 18, 18],
[1332, 50, 172, 144, 197, 137, 59, 72, 40, 98, 162, 36, 145, 173, 99, 85, 45, 81, 40, 198, 101, 40, 99, 80, 48, 22, 88, 102, 50, 62],
[195, 297, 209, 192, 190, 150, 276, 209, 195, 263, 224, 187, 197, 269, 49, 38, 30, 27, 26, 27, 19, 24, 29, 19, 28, 17, 16, 15, 7, 9],
]

optimization_methods = ['SLSQP', 'BFGS', 'L-BFGS-B']

# Dictionary to store convergence information foqqr each method
convergence_info_by_method = {method: {'values': [], 'num_iterations': []} for method in optimization_methods}

for i, method in enumerate(optimization_methods):
    convergence_info_by_method[method]['num_iterations'][:20] = array_of_methods[i]
    num_iterations = convergence_info_by_method[method]['num_iterations'][:20]
    print(num_iterations)
    plt.plot(num_iterations, label=method)

plt.xlabel('Time Step')
plt.ylabel('Number of Iterations')
plt.title('Number of Iterations Comparison of Optimization Methods')
plt.legend()
plt.show()
input("Press Enter to continue...")