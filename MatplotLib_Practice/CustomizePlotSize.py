import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a figure with specific size (width, height)
plt.figure(figsize=(10, 2))  # width=10 inches, height=5 inches

# Plot data
plt.plot(x, y)

# Add title and labels
plt.title('Plot with Custom Size')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# Show plot
plt.show()
