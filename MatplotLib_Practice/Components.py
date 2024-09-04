import matplotlib.pyplot as plt

#Create figure and axis objects
fig, ax = plt.subplots()


#Sample data
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

#Plot data
ax.plot(x, y)

#Set titles and labels
ax.set_title("Square numbers")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")

#show the plot
plt.show()