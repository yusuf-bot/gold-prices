import csv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tabulate import tabulate

# Load data from CSV file
data = np.genfromtxt("goldstock.csv", delimiter=",", skip_header=1, usecols=(1, 2), dtype=float)

num_data = data.shape[0]
labels = np.full(num_data, -1, dtype=int)

# Define price movement states
count = {
    0: "dec > 5.00%",
    1: "dec >2.00%",
    2: "dec >1.75%",
    3: "dec >1.50%",
    4: "dec >1.25%",
    5: "dec >1.00%",
    6: "dec >0.75%",
    7: "dec >0.50%",
    8: "dec >0.25%",
    9: "dec <0.25%",
    10: "inc <0.25%",
    11: "inc >0.25%",
    12: "inc >0.50%",
    13: "inc >0.75%",
    14: "inc >1.00%",
    15: "inc >1.25%",
    16: "inc >1.50%",
    17: "inc >1.75%",
    18: "inc >2.00%",
    19: "inc >5.00%"
}
curr_price = data[0, 1]
price_changes = data[:, 1] / curr_price

# Define conditions for labeling
conditions = [
    (price_changes < 0.95),
    (price_changes > 0.95) & (price_changes <= 0.98),
    (price_changes > 0.98) & (price_changes <= 0.9825),
    (price_changes > 0.9825) & (price_changes <= 0.985),
    (price_changes > 0.985) & (price_changes <= 0.9875),
    (price_changes > 0.9875) & (price_changes <= 0.99),
    (price_changes > 0.99) & (price_changes <= 0.9925),
    (price_changes > 0.9925) & (price_changes <= 0.995),
    (price_changes > 0.995) & (price_changes <= 0.9975),
    (price_changes > 0.9975) & (price_changes < 1),
    (price_changes >= 1) & (price_changes < 1.0025),
    (price_changes >= 1.0025) & (price_changes < 1.005),
    (price_changes >= 1.005) & (price_changes < 1.0075),
    (price_changes >= 1.0075) & (price_changes < 1.01),
    (price_changes >= 1.01) & (price_changes < 1.0125),
    (price_changes >= 1.0125) & (price_changes < 1.015),
    (price_changes >= 1.015) & (price_changes < 1.0175),
    (price_changes >= 1.0175) & (price_changes < 1.02),
    (price_changes >= 1.02) & (price_changes <= 1.05),
    (price_changes > 1.05)
]

# Label the price changes based on the defined conditions
for i, condition in enumerate(conditions):
    labels[condition] = i

# Debugging: Print unique labels to check for out-of-bounds indices
unique_labels = np.unique(labels)
print("Unique Labels:", unique_labels)

# Initialize the transition matrix
transition_matrix = np.zeros((len(count), len(count)))

# Populate the transition matrix
for i in range(num_data - 1):
    if labels[i] != -1 and labels[i + 1] != -1:
        transition_matrix[labels[i], labels[i + 1]] += 1

# Normalize the transition matrix
for i in range(len(count)):
    if np.sum(transition_matrix[i]) > 0:
        transition_matrix[i] /= np.sum(transition_matrix[i])

# Print transition probabilities
print("Transition Probabilities:")

tab=[]
for i, state1 in enumerate(count.values()):
    max_prob = -1
    max_state = None
    for j, state2 in enumerate(count.values()):
        if transition_matrix[i, j] > max_prob:
            max_prob = transition_matrix[i, j]
            max_state = state2

    tab.append([state1,max_state,f"{max_prob:.4f}"])

print(tabulate(tab, headers=["From State", "To State", "Probability"], tablefmt="grid"))

# Create a directed graph for visualization
G = nx.DiGraph()

# Add nodes for each state
for state in count.values():
    G.add_node(state)

# Add edges based on the transition probabilities
for i, state1 in enumerate(count.values()):
    for j, state2 in enumerate(count.values()):
        if transition_matrix[i, j] > 0:
            G.add_edge(state1, state2, weight=transition_matrix[i, j])

# Prepare edge labels for visualization
edge_labels = nx.get_edge_attributes(G, 'weight')
edge_labels = {(u, v): f"{w:.2f}" for (u, v, w) in G.edges(data='weight')}

# Set the layout for the graph
pos = nx.spring_layout(G, seed=42,k=5.4)

# Create the plot
plt.figure(figsize=(18, 15))

# Define node colors based on state type
node_color = ['red' if 'dec' in node else 'green' for node in G.nodes()]
edge_colors = [G[u][v]['weight'] for u, v in G.edges()]

# Draw the nodes, edges, and labels
nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=2700, alpha=0.8, linewidths=2, edgecolors='black')
nx.draw_networkx_edges(G, pos, width=[G[u][v]['weight'] * 10 for u, v in G.edges()], edge_color=edge_colors, alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5)

plt.axis('off')  # Turn off the axis
plt.title("Transition Probabilities of Gold Price Movements")
plt.show()


# Define indices for decrease and increase states based on the count dictionary
decrease_states = list(range(10))  # Indices for dec states (0 to 9)
increase_states = list(range(10, 20))  # Indices for inc states (10 to 19)

# Initialize dictionaries to store the highest probabilities


dec_to_inc_probs = []
inc_to_dec_probs = []


# Check transitions from decrease states to increase states
for dec_state in decrease_states:
    max_prob = 0
    max_inc_state = None
    for inc_state in increase_states:
        prob = transition_matrix[dec_state, inc_state]
        if prob > max_prob:
            max_prob = prob
            max_inc_state = inc_state
    if max_prob>0:
        dec_to_inc_probs.append((count[dec_state], count[max_inc_state], f"{max_prob:.4f}"))

# Check transitions from increase states to decrease states
for inc_state in increase_states:
    max_prob = 0
    max_dec_state = None
    for dec_state in decrease_states:
        prob = transition_matrix[inc_state, dec_state]
        if prob > max_prob:
            max_prob = prob
            max_dec_state = dec_state
    if max_prob>0:
        inc_to_dec_probs.append((count[inc_state], count[max_dec_state], f"{max_prob:.4f}"))

# Print results for decrease to increase transitions
dec_to_inc_probs.sort(key=lambda x: x[2], reverse=True)
inc_to_dec_probs.sort(key=lambda x: x[2], reverse=True)
print("\n" + "-" * 50 + "\n")
# Print sorted results for decrease to increase transitions
print("Highest Transition Probabilities from Decrease to Increase States (Sorted):")
print(tabulate(dec_to_inc_probs, headers=["Decrease State", "Increase State", "Probability"], tablefmt="grid"))

print("\n" + "-" * 50 + "\n")

# Print sorted results for increase to decrease transitions
print("Highest Transition Probabilities from Increase to Decrease States (Sorted):")
print(tabulate(inc_to_dec_probs, headers=["Increase State", "Decrease State", "Probability"], tablefmt="grid"))
