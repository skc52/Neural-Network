import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

axs[0].plot(x, y1, 'r')
axs[0].set_title('Sine Function')

axs[1].plot(x, y2, 'g')
axs[1].set_title('Cosine Function')

# Add labels
for ax in axs:
    ax.set(ylabel='Y Axis')

axs[1].set_xlabel('X Axis')  # Only set the x-label for the bottom subplot

plt.tight_layout()
plt.show()
