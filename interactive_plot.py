import matplotlib.pyplot as plt
import numpy as np
import mplcursors

class InteractivePlot:
    def __init__(self):
        self.fig, self.ax1 = plt.subplots()
        self.forces = []
        self.angles = []
        self.cursor = mplcursors.cursor(hover=True)
        self.cursor.connect("add", self.update_annotation)
        # fix the y-axis scale for force
        self.ax1.set_ylim(-100, 100)
        # fix the y-axis scale for angle
        self.ax1.set_ylim(-np.pi, np.pi)
        
        

    def update_annotation(self, sel):
        index = sel.target.index
        sel.annotation.set_text(f"Force: {self.forces[index]:.2f}\nAngle: {self.angles[index]:.2f}")

    def update_plot(self, time_step, force, angle):
        self.forces.append(force)
        self.angles.append(angle)

        time_values = np.arange(0, time_step + 0.01, 0.01)
        self.ax1.clear()

        color = 'tab:red'
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Force', color=color)
        self.ax1.plot(time_values, self.forces, color=color)
        self.ax1.tick_params(axis='y', labelcolor=color)

        ax2 = self.ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Angle', color=color)
        ax2.plot(time_values, self.angles, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Force and Angle vs Time')
        plt.show(block=False)
        plt.pause(0.01)  # Adjust the pause duration as needed
