import numpy as np
import matplotlib.pyplot as plt

#Generate random data between 0 and 1

x = np.random.rand(50)

y = np.random.rand(50)


#Create a scatter plot
plt.scatter(x, y, marker='x')


#Add title and labels

plt.title('Scatter Plot of Random Data')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

#Show plot
plt.show()
