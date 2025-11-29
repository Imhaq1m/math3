import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import eig
import matplotlib.pyplot as plt

# 1. Load Data
stations = pd.read_csv("Station.csv")
attractions = pd.read_csv("Attraction.csv")
routes = pd.read_csv("Network.csv")

# 2. Test: Calculate "Raw Appeal" for one station
# Group attractions by station and sum their scores
attractions["Raw_Score"] = attractions["Count_Reviews"] * attractions["Rating"]
station_scores = attractions.groupby("Station_ID")["Raw_Score"].sum()

print("Top 5 Most Attractive Stations (Raw Data):")
print(station_scores.sort_values(ascending=False).head(5))

# 3. Test: Build the Network Graph
G = nx.from_pandas_edgelist(routes, "From_ID", "To_ID", ["Time", "Fare"])

print(
    f"\nNetwork successfully built with {G.number_of_nodes()} stations and {
        G.number_of_edges()
    } connections."
)


# 1. Apply Log Normalization
# We add +1 so we don't try to log(0) if a station has 0 reviews
station_scores_normalized = np.log(station_scores + 1)

# 2. Scale it to a 1-10 range (Optional, but easier to understand)
min_score = station_scores_normalized.min()
max_score = station_scores_normalized.max()

# Formula: (x - min) / (max - min) * 9 + 1
attractiveness_final = (
    (station_scores_normalized - min_score) / (max_score - min_score)
) * 9 + 1

print("\nAdjusted Attractiveness Scores (1-10 Scale):")
print(attractiveness_final.sort_values(ascending=False).head(5))

# 1. Define the weight (e.g., 1 minute = RM 0.20 value? Or just use raw minutes?)
# Let's assume 'weight' in the graph is just Travel Time for now.
# If your CSV has a column 'Total_Cost' (Time + Fare), use that instead.
#
# Calculate shortest paths for ALL pairs (This solves the graph)
path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="Time"))

# Example: Check cost from KLCC (KJ10) to Pasar Seni (KJ14)
start = "KJ10"
end = "KJ14"
cost = path_lengths[start][end]

print(f"\nCalculated Cost from {start} to {end}: {cost} minutes")

# ==========================================
# 4. Build the Transition Matrix (The "Rulebook")
# ==========================================

# Get list of all station IDs
all_stations = station_scores.index.tolist()
n_stations = len(all_stations)

# Create an empty matrix (DataFrame) filled with zeros
# Rows = From, Columns = To
P = pd.DataFrame(0.0, index=all_stations, columns=all_stations)

print("Building Transition Matrix... (This might take a moment)")

# Loop through every "From" station
for i in all_stations:
    # Loop through every "To" station
    for j in all_stations:
        # Get Attractiveness of the Destination (j)
        # Use a default of 1.0 if the station has no specific data
        attractiveness_j = attractiveness_final.get(j, 1.0)

        if i == j:
            # Case: Staying at the same station (Self-loop)
            # We assign a small "virtual cost" (e.g., 2 mins) to represent the "cost" of not moving
            # High attractiveness encourages staying, but we need a non-zero denominator
            likelihood = attractiveness_j / 2.0
        else:
            # Case: Moving from i to j
            # Get cost (Time) from our Dijkstra path dictionary
            # If no path exists (unreachable), cost is infinite (likelihood = 0)
            if j in path_lengths[i]:
                cost_ij = path_lengths[i][j]
                if cost_ij == 0:
                    cost_ij = 1.0  # Prevent divide by zero

                # GRAVITY MODEL FORMULA:
                likelihood = attractiveness_j / cost_ij
            else:
                likelihood = 0

        # Fill the matrix
        P.loc[i, j] = likelihood

# ==========================================
# 5. Normalize (Make Rows Sum to 1)
# ==========================================

# Divide every value by the sum of its row
# This turns "Likelihood Scores" into "Probabilities"
P_normalized = P.div(P.sum(axis=1), axis=0)

# Handle any rows that sum to 0 (dead ends) by making them stay put (1.0 probability to self)
P_normalized = P_normalized.fillna(0)
for i in P_normalized.index:
    if P_normalized.loc[i].sum() == 0:
        P_normalized.loc[i, i] = 1.0

print("Transition Matrix built and normalized.")

# ==========================================
# 6. Solve for Steady State (Linear Algebra)
# ==========================================
# We solve Ax = x (where A is the Transpose of P)

# Transpose P to get A
A = P_normalized.T.values

# Calculate Eigenvectors using scipy (more stable than numpy for this)
# We want the eigenvector associated with the eigenvalue of 1
eigenvalues, eigenvectors = eig(A)

# Find the index where the eigenvalue is closest to 1
idx = np.argmin(np.abs(eigenvalues - 1))
steady_state_vector = np.real(eigenvectors[:, idx])

# Normalize the vector so it sums to 1 (Probabilities must sum to 100%)
steady_state_vector = steady_state_vector / steady_state_vector.sum()

# ==========================================
# 7. Final Output: The Ranking
# ==========================================
final_ranking = pd.Series(steady_state_vector, index=all_stations).sort_values(
    ascending=False
)

print("\n=== FINAL PREDICTION: Top 10 Tourist Hubs ===")
# We assume you have a 'Station_Name' in your stations CSV to map IDs to Names
# If not, just print the IDs
try:
    # Create a map of ID -> Name
    id_to_name = stations.set_index("Station_ID")["Station_Name"].to_dict()
    # Map the index
    final_ranking.index = final_ranking.index.map(
        lambda x: f"{x} ({id_to_name.get(x, 'Unknown')})"
    )
except:
    pass

print(final_ranking.head(10))

# Plot the Top 10
top_10 = final_ranking.head(10)
plt.figure(figsize=(10, 6))
top_10.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Predicted Tourist Distribution (Steady State Probabilities)", fontsize=14)
plt.ylabel("Probability", fontsize=12)
plt.xlabel("Station", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
