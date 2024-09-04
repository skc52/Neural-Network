import matplotlib.pyplot as plt

# Create a figure and a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Sample data
x = [0, 1, 2, 3, 4, 5]
y1 = [0, 1, 4, 9, 16, 25]
y2 = [0, 1, 8, 27, 64, 125]
y3 = [0, 2, 6, 12, 20, 30]
y4 = [5, 4, 3, 2, 1, 0]

# Plot different data on each subplot
axs[0, 0].plot(x, y1, 'r')  # Top-left subplot
axs[0, 1].plot(x, y2, 'g')  # Top-right subplot
axs[1, 0].plot(x, y3, 'b')  # Bottom-left subplot
axs[1, 1].plot(x, y4, 'k')  # Bottom-right subplot

# Set individual titles
axs[0, 0].set_title("y = x^2")
axs[0, 1].set_title("y = x^3")
axs[1, 0].set_title("y = 2x")
axs[1, 1].set_title("y = 5 - x")

# Add a title for the entire figure
fig.suptitle("Multiple Subplots Example")

# Automatically adjust layout
fig.tight_layout()

# Show the plot
plt.show()
