import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)

# Customize x-axis and y-axis ticks

# ticks=np.arange(0, 11, 2):
#  Creates an array of tick positions starting from 0 up to 10 with a step size of 2 (i.e., [0, 2, 4, 6, 8, 10]).
plt.xticks(ticks=np.arange(0, 11, 2), labels=['0', '2', '4', '6', '8', '10'])

# ticks=np.linspace(-1, 1, 5): 
# Generates 5 evenly spaced tick positions between -1 and 1 (i.e., [-1, -0.5, 0, 0.5, 1]).
plt.yticks(ticks=np.linspace(-1, 1, 5), labels=['-1', '-0.5', '0', '0.5', '1'])

plt.title('Plot with Custom Ticks')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

plt.show()
