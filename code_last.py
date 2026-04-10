import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx

# =========================
# 1. Choix du graphe
# =========================

# Choisis un graphe :
#G = nx.barabasi_albert_graph(200, 3)
#G = nx.erdos_renyi_graph(200, 0.02)
G = nx.random_geometric_graph(200, 0.1)

nodes = list(G.nodes)
start = nodes[0]
goal = nodes[-1]

print("Start :", start)
print("Goal :", goal)

# =========================
# 2. Paramètres Q-learning
# =========================

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 500

# Q-table (dictionnaire)
q_table = {}

rewards_per_episode = []

# =========================
# 3. Q-learning
# =========================

for ep in range(episodes):
    state = start
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 1000:  # éviter boucle infinie
        neighbors = list(G.neighbors(state))
        
        # epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            next_state = random.choice(neighbors)
        else:
            qs = [q_table.get((state, n), 0) for n in neighbors]
            next_state = neighbors[np.argmax(qs)]
        
        # reward
        if next_state == goal:
            r = 100
            done = True
        else:
            r = -1
        
        # max Q suivant
        next_neighbors = list(G.neighbors(next_state))
        max_q_next = max([q_table.get((next_state, n), 0) for n in next_neighbors], default=0)
        
        # mise à jour Q
        old_q = q_table.get((state, next_state), 0)
        q_table[(state, next_state)] = old_q + alpha * (r + gamma * max_q_next - old_q)
        
        state = next_state
        total_reward += r
        steps += 1
    
    rewards_per_episode.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# =========================
# 4. Courbe de convergence
# =========================

plt.plot(rewards_per_episode)
plt.xlabel("Épisodes")
plt.ylabel("Récompense cumulée")
plt.title("Convergence du Q-learning (graphe)")
plt.show()

# =========================
# 5. Extraction du chemin
# =========================

state = start
path = [state]
visited = set()

for _ in range(1000):
    if state == goal:
        break
        
    neighbors = list(G.neighbors(state))
    qs = [q_table.get((state, n), 0) for n in neighbors]
    
    next_state = neighbors[np.argmax(qs)]
    
    # éviter boucle
    if next_state in visited:
        break
        
    path.append(next_state)
    visited.add(next_state)
    state = next_state

print("Chemin trouvé :", path)

# =========================
# 6. Visualisation du graphe
# =========================

plt.figure(figsize=(8, 8))

pos = nx.spring_layout(G)

nx.draw(G, pos, node_size=20, alpha=0.5)

# chemin en rouge
edges_in_path = list(zip(path, path[1:]))
nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color='r', width=2)

# start et goal
nx.draw_networkx_nodes(G, pos, nodelist=[start], node_color='green', node_size=100)
nx.draw_networkx_nodes(G, pos, nodelist=[goal], node_color='blue', node_size=100)

plt.title("Chemin appris par Q-learning")
plt.show()