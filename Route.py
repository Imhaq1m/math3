import pandas as pd
import numpy as np
import re

# 1. Load the dataset
df = pd.read_csv("Route.csv")

# Rename the first column to 'Origin' (it was likely Unnamed in the CSV)
df.rename(columns={df.columns[0]: "Origin"}, inplace=True)

# 2. Reshape from Wide to Long format using Melt
# This creates a row for every Origin-Destination pair
df_long = df.melt(id_vars=["Origin"],
                  var_name="Destination", value_name="Route_String")


# 3. Define a parsing function to extract details from the Route_String
def parse_route_info(route_str):
    if pd.isna(route_str):
        return 0, [], []

    # Split the route into segments based on ' >> '
    segments = route_str.split(" >> ")

    # Calculate number of transfers
    # If there is 1 segment, transfers = 0. If 2 segments, transfers = 1.
    num_transfers = len(segments) - 1

    lines = []
    transfer_stations = []

    # Regex pattern to capture: LineName[Start > End]
    pattern = re.compile(r"([A-Z]+)\[(.*?)\s>\s(.*?)\]")

    for i, segment in enumerate(segments):
        match = pattern.search(segment)
        if match:
            line_code = match.group(1)
            start_station = match.group(2)
            end_station = match.group(3)

            lines.append(line_code)

            # The end station of the current segment is the transfer point
            # (unless it's the very last segment of the journey)
            if i < num_transfers:
                transfer_stations.append(end_station)

    return num_transfers, lines, transfer_stations


# 4. Apply the parsing function
# We use zip to unpack the results into new columns efficiently
parsed_data = [parse_route_info(r) for r in df_long["Route_String"]]
df_long["Num_Transfers"], df_long["Lines_Used"], df_long["Transfer_Stations"] = zip(
    *parsed_data
)

# 5. (Optional) Filter out rows where Origin == Destination if needed
df_long = df_long[df_long["Origin"] != df_long["Destination"]]

# Display the first few rows
print(df_long.head())
print(df_long[df_long["Num_Transfers"] == 1])
