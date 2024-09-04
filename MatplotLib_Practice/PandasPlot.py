import pandas as pd
import matplotlib.pyplot as plt


#Create a simple dataframe

data = {
    'A': [1, 2, 3, 4, 5],
    'B': [2, 3, 4, 5, 6],
    'C': [5, 6, 7, 8, 9]
}

df = pd.DataFrame(data)


# df.plot()


# Plotting Specific Columns
# df['A'].plot(label='Column A', linestyle='--', marker='o')
# df['B'].plot(label='Column B', linestyle='-', marker='x')


# Plotting a bar plot
df.plot(kind='bar')
plt.title('Bar Plot of DataFrame Columns')
plt.xlabel('Index')
plt.ylabel('Values')
plt.show()



