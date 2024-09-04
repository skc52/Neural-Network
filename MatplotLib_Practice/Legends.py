import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y1 = [2, 4, 6, 8, 10]
y2 = [1, 3, 5, 7, 9]


#Create line plots

plt.plot(x, y1, label='Line 1', color = 'blue', marker = 'o')
plt.plot(x, y2, label = 'Line 2', color = 'red', linestyle = '--', marker='x')

# Add title and labels
plt.title('Line Plot with Legends')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

#Add legend
plt.legend()

# Show plot
plt.show()


