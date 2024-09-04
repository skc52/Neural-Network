import matplotlib.pyplot as plt 

#sample data
categories = ['A', 'B', 'C', 'D']
values = [3, 7, 2, 5]

#create a bar plot
plt.bar(categories, values)

#Add title("Bar Plot Example")
plt.title("Bar Plot Example")
plt.xlabel("Categories")
plt.ylabel("Values")

#Show the plot
plt.show()


