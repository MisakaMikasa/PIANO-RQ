import random
from collections import deque

def bfs_sample(input_file, output_file, start_node=29, max_nodes=500):
    # Read the input adjacency list
    adj_list = {}
    with open(input_file, 'r') as file:
        for line in file:
            u, v = map(int, line.strip().split())
            if u not in adj_list:
                adj_list[u] = []
            adj_list[u].append(v)

    # BFS Sampling
    visited = set()
    sampled_nodes = []
    queue = deque([start_node])
    
    while queue and len(sampled_nodes) < max_nodes:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            sampled_nodes.append(node)
            if node in adj_list:  # Add neighbors to the queue
                queue.extend(adj_list[node])

    # Build the subgraph
    subgraph_edges = []
    sampled_set = set(sampled_nodes)
    for node in sampled_nodes:
        if node in adj_list:
            for neighbor in adj_list[node]:
                if neighbor in sampled_set:
                    # Generate a random weight between 0 and 1
                    weight = round(random.uniform(0, 0.75), 2)
                    subgraph_edges.append((node, neighbor, weight))
    
    # Re-number nodes starting from 0
    node_mapping = {node: i for i, node in enumerate(sampled_nodes)}
    renumbered_edges = [(node_mapping[u], node_mapping[v], w) for u, v, w in subgraph_edges]
    
    final_edges = []
    adjacency_count = {u: 0 for u in range(len(sampled_nodes))}
    
    # Count initial degrees
    for u, v, _ in renumbered_edges:
        adjacency_count[u] += 1

    for u, v, w in renumbered_edges:
        # Remove edge with 75% chance if it would not make adj[u] empty
        if random.random() < 0.75 and adjacency_count[u] > 1:
            adjacency_count[u] -= 1
        else:
            final_edges.append((u, v, w))

    # Write the output adjacency list
    with open(output_file, 'w') as file:
        for u, v, w in final_edges:
            file.write(f"{u} {v} {w}\n")

# Example usage:
bfs_sample('C:\\Users\\17789\\Desktop\\Graph Dataset\\wiki-Vote.txt', 'C:\\Users\\17789\\Desktop\\Graph Dataset\\subgraph1.txt')
