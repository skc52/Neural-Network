import numpy as np
import matplotlib.pyplot as plt

# Create data using NumPy
x = np.linspace(0, 10, 100)  # 100 points from 0 to 10
y = np.sin(x)  # Sine function

# Create a line plot
plt.plot(x, y)

# Add title and labels
plt.title('Sine Wave')
plt.xlabel('X Axis')
plt.ylabel('sin(X)')

# Show plot
plt.show()
