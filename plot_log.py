import matplotlib.pyplot as plt
import os

def plot_loss(log_file_path, limit=None):
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    if limit:
        loss = [float(line) for line in lines[:limit]]
    else:
        loss = [float(line) for line in lines]
    plt.figure()
    plt.plot(loss)
    plt.title('Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
def plot_input(log_file_path, limit=None):
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    if limit:
        inputs = [float(line) for line in lines[:limit]]
    else:
        inputs = [float(line) for line in lines]    
    plt.figure()
    plt.plot(inputs)
    plt.title('Input')
    plt.xlabel('Iteration')
    plt.ylabel('Input')
    
def plot_angle(log_file_path, limit=None):
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    if limit:
        angles = [float(line) for line in lines[:limit]]
    else:
        angles = [float(line) for line in lines]    
    plt.figure()
    plt.plot(angles)
    plt.title('Angle')
    plt.xlabel('Iteration')
    plt.ylabel('Angle')
    
def plot_pos(log_file_path, limit=None):
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    if limit:
        positions = [float(line) for line in lines[:limit]]
    else:
        positions = [float(line) for line in lines]    
    plt.figure()
    plt.plot(positions)
    plt.title('Position')
    plt.xlabel('Iteration')
    plt.ylabel('Position')

def plot_time(log_file_path, limit=None):
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    if limit:
        times = [float(line) for line in lines[:limit]]
    else:
        times = [float(line) for line in lines]
    plt.figure()
    plt.plot(times)
    plt.title('Time')
    plt.xlabel('Iteration')
    plt.ylabel('Time')
    
    
# Example usage
# read all txt files in the current directory containing "slsqp" in their name and number 10
solver = 'slsqp'
num = 20
loss_log_file_path = [f for f in os.listdir() if f.endswith(f"{solver}_{num}.txt") and f.startswith('loss')][0]
input_log_file_path = [f for f in os.listdir() if f.endswith(f"{solver}_{num}.txt") and f.startswith('input')][0]
angle_log_file_path = [f for f in os.listdir() if f.endswith(f"{solver}_{num}.txt") and f.startswith('angle')][0]
pos_log_file_path = [f for f in os.listdir() if f.endswith(f"{solver}_{num}.txt") and f.startswith('pos')][0]
time_log_file_path = [f for f in os.listdir() if f.endswith(f"{solver}_{num}.txt") and f.startswith('time')][0]


plot_loss(loss_log_file_path,limit=100)
plot_input(input_log_file_path,limit=100)
plot_angle(angle_log_file_path,limit=100)
plot_pos(pos_log_file_path,limit=100)
plot_time(time_log_file_path,limit=100)

plt.show()