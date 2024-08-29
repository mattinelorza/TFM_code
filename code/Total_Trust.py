import pandas as pd
import matplotlib.pyplot as plt

# Importar .py
import dataset_fuzzy
import dataset2_fuzzy

# Coger dataframes de datasets
df1 = dataset_fuzzy.df
df2 = dataset2_fuzzy.df

# 'Network_Trust' por 0.5 df1
df1['Network_Trust'] = df1['Network_Trust'] * 0.5

# 'node_reputation' por 0.5 df2
df2['node_reputation'] = df2['node_reputation'] * 0.5


print("Updated dataframe from dataset_fuzzy.py:")
print(df1)

print("\nUpdated dataframe from dataset2_fuzzy.py:")
print(df2)

combined_df = pd.concat([df1, df2], axis=1)

combined_df['Total_Trust'] = combined_df['Network_Trust'] + combined_df['node_reputation']


# DF actualizado
print("Updated dataframe with Total_Trust:")
print(combined_df)


# Plot Network_Trust puntos
plt.scatter(combined_df.index, combined_df['Network_Trust'], label='Network_Trust', marker='o', edgecolor='black')

# Plot node_reputation puntos
plt.scatter(combined_df.index, combined_df['node_reputation'], label='Node_Reputation',  marker='o', edgecolor='black')

#Plot Total_Trust with lineas
plt.plot(combined_df.index, combined_df['Total_Trust'], label='Total_Trust', color='green', linestyle='-', marker='o')

plt.title('Total_Trust (Network_Trust x 0.5 + Node_Reputation x 0.5)')
plt.xlabel('Data')
plt.ylabel('Total_Trust')
plt.legend()
plt.grid(True)
plt.show()




