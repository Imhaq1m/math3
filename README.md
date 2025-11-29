Modeling Tourist Movement in Kuala Lumpur 🇲🇾

This project uses Markov Chains and a Gravity Model to analyze tourist movement patterns across the RapidKL train network (LRT, MRT, Monorail). It predicts the most critical "Tourist Hubs" based on station attractiveness and network connectivity.

📂 Files in this Repository

tourist_markov_model.py: The main Python script.

Station.csv: List of stations and their names.

Attraction.csv: Tourist spots, reviews, ratings, and accessibility penalties.

Network.csv: Travel times and fares between stations.

🚀 How to Run the Code

1. Prerequisite: Install Python

Make sure you have Python installed. If not, download it from python.org.

2. Download the Project

Clone this repository or download the ZIP file and extract it.

git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)


3. Install Libraries

pip install pandas numpy networkx scipy matplotlib


4. Run the Model

Execute the script:

python Model.py


📊 Output

The script will:

Load the data and calculate attractiveness scores.

Build the transport network graph.

Simulate tourist movement (Markov Chain).

Print the Top 10 Tourist Hubs in the terminal.

Pop up a Bar Chart visualizing the steady-state probabilities.

🧠 Methodology

Gravity Model: Likelihood = Attractiveness / Generalized_Cost

Cost Function: Time + (Fare * 5 mins).

Steady State: Solved using Eigenvector Centrality ($Ax = x$).
