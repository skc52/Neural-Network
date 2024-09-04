import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

#Create a Figure and GridSpec Object

fig = plt.figure(figsize=(10, 8))

#Define a GridSpec object: 3 rows and 3 columns
gs = GridSpec(3, 3)


# Create subplots based on GridSpec layout
ax1 = plt.subplot(gs[0, :])  # First row, all columns
ax2 = plt.subplot(gs[1, :-1])  # Second row, first two columns
ax3 = plt.subplot(gs[1:, -1])  # Last column, second and third rows
ax4 = plt.subplot(gs[2, 0])  # Last row, first column
ax5 = plt.subplot(gs[2, 1])  # Last row, second column


x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = np.exp(x / 10)
y5 = np.log(x + 1)

# Plotting data
ax1.plot(x, y1, 'r')
ax1.set_title('Sine Function')

ax2.plot(x, y2, 'g')
ax2.set_title('Cosine Function')

ax3.plot(x, y3, 'b')
ax3.set_title('Tangent Function')

ax4.plot(x, y4, 'm')
ax4.set_title('Exponential Function')

ax5.plot(x, y5, 'c')
ax5.set_title('Logarithmic Function')


plt.tight_layout()
plt.show()
