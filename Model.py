import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import eig
import matplotlib.pyplot as plt

# ==========================================
# 1. Load Data
# ==========================================
print("Loading data...")
try:
    stations = pd.read_csv("Station.csv")
    attractions = pd.read_csv("Attraction.csv")
    routes = pd.read_csv("Network.csv")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print(
        "Please ensure Station.csv, Attraction.csv, and Network.csv are in the folder."
    )
    exit()

# ==========================================
# 2. Process Attractiveness (With Penalty)
# ==========================================
print("Calculating Attractiveness Scores...")

# Handle missing 'Last_Mile_Difficulty' by defaulting to 1.0 (Easy/Walk) if column is missing
if "Last_Mile_Difficulty" not in attractions.columns:
    attractions["Last_Mile_Difficulty"] = 1.0

# --- NEW LOGIC: PENALIZE HARD-TO-REACH SPOTS ---
# Formula: (Reviews * Stars) / Penalty
# Penalty 1.0 = Walkable (Score stays same)
# Penalty 2.0 = Needs Bus/Taxi (Score is halved)
attractions["Raw_Score"] = (
    attractions["Count_Reviews"] * attractions["Rating"]
) / attractions["Last_Mile_Difficulty"]

# Sum up the scores per station (in case a station has multiple attractions)
station_scores = attractions.groupby("Station_ID")["Raw_Score"].sum()

# Print top 5 raw scores to verify
print("\nTop 5 Raw Scores (Pre-Normalization):")
print(station_scores.sort_values(ascending=False).head(5))

# Log Normalization: Fixes the issue where 50,000 reviews overpowers everything
# We add +1 to avoid log(0) errors
station_scores_normalized = np.log(station_scores + 1)

# Scale to a 1-10 range for easier interpretation in the Gravity Model
min_score = station_scores_normalized.min()
max_score = station_scores_normalized.max()
attractiveness_final = (
    (station_scores_normalized - min_score) / (max_score - min_score)
) * 9 + 1

print("\nTop 5 Adjusted Attractiveness Scores (1-10 Scale):")
print(attractiveness_final.sort_values(ascending=False).head(5))

# ==========================================
# 3. Build Network (With Generalized Cost)
# ==========================================
print("\nBuilding Transport Network...")

# --- NEW OPTIMIZATION LOGIC ---
# Define the "Value of Time": 1 Ringgit = 5 Minutes of perceived pain
TIME_VALUE_OF_MONEY = 5.0

# Create Generalized Cost = Time + (Fare * 5)
routes["Generalized_Cost"] = routes["Time"] + (routes["Fare"] * TIME_VALUE_OF_MONEY)

# Build the Graph using NetworkX
# We use 'Generalized_Cost' as the weight for pathfinding
G = nx.from_pandas_edgelist(routes, "From_ID", "To_ID", ["Generalized_Cost"])

print(
    f"Graph built with {G.number_of_nodes()} stations and {
        G.number_of_edges()
    } connections."
)

# Calculate shortest paths for ALL pairs using Dijkstra
# This gives us the 'Cost' denominator for every possible trip
path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="Generalized_Cost"))

# Test: Check cost between two famous stations
try:
    start_test, end_test = "KJ10", "KJ14"  # KLCC to Pasar Seni
    if start_test in path_lengths and end_test in path_lengths[start_test]:
        print(
            f"Test Cost ({start_test} -> {end_test}): {
                path_lengths[start_test][end_test]:.2f} units (Time + Fare penalty)"
        )
except:
    pass

# ==========================================
# 4. Build Transition Matrix (Gravity Model)
# ==========================================
print("Building Transition Matrix (The Rulebook)...")

all_stations = station_scores.index.tolist()
P = pd.DataFrame(0.0, index=all_stations, columns=all_stations)

for i in all_stations:
    for j in all_stations:
        # Get Destination Attractiveness (Default to 1.0 if missing)
        attractiveness_j = attractiveness_final.get(j, 1.0)

        if i == j:
            # Self-loop: "Cost" of staying is low (e.g., 2 units)
            # High attractiveness encourages staying
            likelihood = attractiveness_j / 2.0
        else:
            # Movement: Check if a path exists
            if i in path_lengths and j in path_lengths[i]:
                cost_ij = path_lengths[i][j]
                if cost_ij <= 0:
                    cost_ij = 0.1  # Safety valve for zero cost

                # GRAVITY MODEL: P ~ Attractiveness / Cost
                likelihood = attractiveness_j / cost_ij
            else:
                likelihood = 0.0

        P.loc[i, j] = likelihood

# ==========================================
# 5. Normalize (Row Stochastic)
# ==========================================
# Divide every value by the sum of its row so probabilities sum to 1.0
P_normalized = P.div(P.sum(axis=1), axis=0)

# Handle dead ends (rows summing to 0) by making them stay put
P_normalized = P_normalized.fillna(0)
for i in P_normalized.index:
    if P_normalized.loc[i].sum() == 0:
        P_normalized.loc[i, i] = 1.0

print("Matrix Normalized.")

# ==========================================
# 6. Solve Steady State (Eigenvector)
# ==========================================
print("Solving for Steady State...")

# Transpose P to get A for the equation Ax = x
A = P_normalized.T.values

# Calculate Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = eig(A)

# Find the eigenvector corresponding to eigenvalue ~1
idx = np.argmin(np.abs(eigenvalues - 1))
steady_state_vector = np.real(eigenvectors[:, idx])

# Normalize vector to sum to 1 (100% Probability)
steady_state_vector = steady_state_vector / steady_state_vector.sum()

# ==========================================
# 7. Final Output & Visualization
# ==========================================
final_ranking = pd.Series(steady_state_vector, index=all_stations).sort_values(
    ascending=False
)

# Map IDs to Names for display
try:
    id_to_name = stations.set_index("Station_ID")["Station_Name"].to_dict()
    final_ranking.index = final_ranking.index.map(
        lambda x: f"{x} ({id_to_name.get(x, 'Unknown')})"
    )
except:
    print("Warning: Could not map Station IDs to Names. Displaying IDs only.")

print("\n=== FINAL PREDICTION: Top 10 Tourist Hubs ===")
print(final_ranking.head(10))

# Plot
top_10 = final_ranking.head(10)
plt.figure(figsize=(10, 6))
top_10.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Projected Tourist Distribution (Steady State)", fontsize=14)
plt.ylabel("Probability", fontsize=12)
plt.xlabel("Station", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
