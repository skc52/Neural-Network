import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(8, 4))  # Set figure size
ax.plot(x, y)




#set aspect ratio (1 means equal scaling)
ax.set_aspect('2')


# Add grid lines
plt.grid(True)  # or plt.grid(False) to disable
plt.title('Plot with Aspect Ratio 1:1')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

plt.show()