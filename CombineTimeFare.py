import pandas as pd
import numpy as np

# 1. Load the existing files
df_time = pd.read_csv("Time_cleaned.csv")
df_fare = pd.read_csv("Fare_cleaned.csv")

# 2. Extract the columns you want as Numpy arrays
# .values or .to_numpy() converts a pandas Series to a numpy array
time_array = df_time["Time"].to_numpy()
fare_array = df_fare["Fare"].to_numpy()
from_id_array = df_time["From_ID"].to_numpy()
to_id_array = df_time["To_ID"].to_numpy()

# 3. Combine them using np.column_stack
# This stacks the arrays as columns in a new 2D matrix
combined_data = np.column_stack(
    (from_id_array, to_id_array, time_array, fare_array))

# Preview the result (It will be an array of objects since IDs are strings)
print("Combined Array Sample:")
print(combined_data[:5])

# 4. (Optional) Convert back to a DataFrame to save nicely
df_final = pd.DataFrame(combined_data, columns=[
                        "From_ID", "To_ID", "Time", "Fare"])
df_final.to_csv("Network.csv", index=False)
