import matplotlib.pyplot as plt 
import numpy as np

#Sample data: generate 1000 random numbers following a normal distribution
data = np.random.randn(100000)


#Create a histogram

# bins means the number of bars in histograms
plt.hist(data, bins=100, edgecolor='black')


# Add a title
plt.title('Histogram of Normally Distributed Data')

# Label the x and y axes
plt.xlabel('Value')
plt.ylabel('Frequency')

# Display the plot
plt.show()