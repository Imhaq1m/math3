import pandas as pd
import numpy as np

df = pd.read_csv("Fare.csv")

# Automatically detect the first column name
first_col = df.columns[0]

# Rename it to "Origin"
df = df.rename(columns={first_col: "Origin"})

# Convert numeric matrix to numpy
fares = df.iloc[:, 1:].to_numpy()

# Set diagonal to 0
np.fill_diagonal(fares, 0)

# Put values back
df.iloc[:, 1:] = fares

# Melt into long format
df_long = df.melt(id_vars="Origin", var_name="Destination", value_name="Fare")

df_long.to_csv("Fare_cleaned_Original.csv", index=False)

print(df_long)
