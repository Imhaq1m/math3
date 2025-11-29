import pandas as pd
import numpy as np

df = pd.read_csv("Time.csv")

# Automatically detect the first column name
first_col = df.columns[0]

# Rename it to "Origin"
df = df.rename(columns={first_col: "Origin"})

# Convert numeric matrix to numpy
fares = df.iloc[:, 1:].to_numpy()

# Melt into long format
df_long = df.melt(id_vars="Origin", var_name="Destination", value_name="Time")

df_long.to_csv("Time_cleaned_Original.csv", index=False)

print(df_long)
