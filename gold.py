import csv
import matplotlib.pyplot as plt
import networkx as nx

import numpy as np

data = np.genfromtxt(
    "goldstock.csv", delimiter=",", skip_header=1, usecols=(1, 2), dtype=float
)

num_data = data.shape[0]
labels = np.full(num_data, -1, dtype=int)

count = {
    0: "dec >2.00%",
    1: "dec >1.75%",
    2: "dec >1.50%",
    3: "dec >1.25%",
    4: "dec >1.00%",
    5: "dec >0.75%",
    6: "dec >0.50%",
    7: "dec >0.25%",
    8: "dec <0.25%",
    9: "inc <0.25%",
    10: "inc >0.25%",
    11: "inc >0.50%",
    12: "inc >0.75%",
    13: "inc >1.00%",
    14: "inc >1.25%",
    15: "inc >1.50%",
    16: "inc >1.75%",
    17: "inc >2.00%",
    18: "inc > 5%",
    19: "dec > 5%",
}

curr_price = data[0, 1]
price_changes = data[:, 1] / curr_price
conditions = [
    (price_changes > 1.02) & (price_changes <= 0.98),
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
    (price_changes > 1.05),
    (price_changes < 0.95),
]

for i, condition in enumerate(conditions):
    labels[condition] = i

transition_matrix = np.zeros((len(count), len(count)))

for i in range(num_data - 1):
    if labels[i] != -1 and labels[i + 1] != -1:
        transition_matrix[labels[i], labels[i + 1]] += 1

for i in range(len(count)):
    if np.sum(transition_matrix[i]) > 0:
        transition_matrix[i] /= np.sum(transition_matrix[i])

print("Transition Probabilities:")
print("-" * 50)
print("From State \t To State \t Probability")
print("-" * 50)


for i, state1 in enumerate(count.values()):
    max_prob = -1
    max_state = None
    for j, state2 in enumerate(count.values()):
        if transition_matrix[i, j] > max_prob:
            max_prob = transition_matrix[i, j]
            max_state = state2

    print(f"{state1:15}   {max_state:15}  {max_prob:.4f}")

print("-" * 50)

G = nx.DiGraph()

for state in count.values():
    G.add_node(state)

for i, state1 in enumerate(count.values()):
    for j, state2 in enumerate(count.values()):
        if transition_matrix[i, j] > 0:
            G.add_edge(state1, state2, weight=transition_matrix[i, j])

edge_labels = nx.get_edge_attributes(G, 'weight')
edge_labels = {(u, v): f"{w:.2f}" for (u, v, w) in G.edges(data='weight')}


# Set the Kamada-Kaway layout with the distance dictionary
pos = nx.spring_layout(G,seed=42,k=5)

plt.figure(figsize=(12, 10))

node_color = ['red' if 'dec' in node else 'green' for node in G.nodes()]
edge_colors = [G[u][v]['weight'] for u, v in G.edges()]

nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=2500, alpha=0.8, linewidths=2,edgecolors='black')
nx.draw_networkx_edges(G, pos, width=edge_colors * 10, edge_color=edge_colors, alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5)

plt.axis('off')
plt.show()           
