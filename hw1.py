import networkx as nx
from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import random

# Parse out the adjacency list from MOLT-4_A.txt
def get_adj_list(file):
    adj_list = []
    with open(file, 'r') as f:
        for line in f:
            row, col = map(int, line.strip().split(','))
            adj_list.append((row, col))
            
    return adj_list

# Parses out the graph data (just a number on each line) into a list
def get_graph_data(file):
    with open(file, 'r') as f:
        data = [int(line.strip()) for line in f]
    return data

# Create the graphs from the label and sort it by labels
def create_graphs_by_label(adj_list, graph_indicators, node_labels, edge_labels, graph_labels):
    graphs = {}  # Outer dictionary: {graph_label: {graph_id: graph}}

    for i, (row, col) in enumerate(adj_list):
        graph_id = graph_indicators[row - 1]
        graph_label = graph_labels[graph_id - 1]  # Get the label for this graph

        # Initialize dictionaries if not present
        if graph_label not in graphs:
            graphs[graph_label] = {}  # Inner dictionary for graphs with this label
        if graph_id not in graphs[graph_label]:
            graphs[graph_label][graph_id] = nx.Graph()
            graphs[graph_label][graph_id].graph['label'] = graph_label

        # Add edge and edge label
        graph = graphs[graph_label][graph_id]
        graph.add_edge(row, col, label=edge_labels[i])

        # Add node labels if not already present
        if 'label' not in graph.nodes[row]:
            graph.nodes[row]['label'] = node_labels[row - 1]
        if 'label' not in graph.nodes[col]:
            graph.nodes[col]['label'] = node_labels[col - 1]

    return graphs

def get_frequencies(G, k):
    counter = Counter()
    
    # Go through every possible subgraph of size k in G
    for nodes in combinations(G.nodes(), k):
        subgraph = G.subgraph(nodes)
        if nx.is_connected(subgraph):
            # Add all node labels
            node_labels = [G.nodes[n]['label'] for n in subgraph.nodes()]
            
            # Add all edge labels and their corresponding node labels
            edge_labels = []
            for u, v, d in subgraph.edges(data=True):
                edge_label = (G.nodes[u]['label'], G.nodes[v]['label'], d['label'])
                edge_labels.append(edge_label)
            
            labels = (tuple(sorted(node_labels)), tuple(sorted(edge_labels)))
            counter[labels] += 1
            
    return counter

def get_graphlet_kernel(G1, G2, k):
    count_G1 = get_frequencies(G1, k)
    count_G2 = get_frequencies(G2, k)

    # Conver to counts to vectors based all possible subgraphs
    all_subgraphs = set(count_G1.keys()).union(count_G2.keys())
    vector_G1 = np.array([count_G1.get(subgraph, 0) for subgraph in all_subgraphs])
    vector_G2 = np.array([count_G2.get(subgraph, 0) for subgraph in all_subgraphs])

    return np.dot(vector_G1, vector_G2)

def get_runtimes(graphs, max_k, sample_size=100):
    runtimes0 = []
    runtimes1 = []
    runtimes_total = []

    # Randomly select 100 graphs from each label
    graphs0_sample = random.sample(list(graphs[0].values()), sample_size)
    graphs1_sample = random.sample(list(graphs[1].values()), sample_size)

    for k in range(1, max_k):
        # For graphs with label 0
        total_time0 = 0
        for graph in graphs0_sample:
            start_time = time.time()
            get_frequencies(graph, k)  
            end_time = time.time()
            total_time0 += (end_time - start_time)
        runtimes0.append(total_time0 / sample_size)
        
        # For graphs with label 1
        total_time1 = 0
        for graph in graphs1_sample:
            start_time = time.time()
            get_frequencies(graph, k)  
            end_time = time.time()
            total_time1 += (end_time - start_time)
        runtimes1.append(total_time1 / sample_size)
        
        runtimes_total.append((total_time1 + total_time0) / (sample_size * 2))
    
    return runtimes0, runtimes1, runtimes_total

def calculate_similarity_for_k(graphs_0, graphs_1, k, num_comparisons=100):
    same_class_similarities0 = []
    same_class_similarities1 = []
    diff_class_similarities = []
    
    # Same class comparisons (within graph0)
    graph0_combinations = list(graphs_0.items())
    random.shuffle(graph0_combinations)
    for i in range(num_comparisons):
        G1 = graph0_combinations[i][1]
        G2 = graph0_combinations[i + 1][1]
        similarity = get_graphlet_kernel(G1, G2, k)
        same_class_similarities0.append(similarity)

    # Same class comparisons (within graph1)
    graph1_combinations = list(graphs_1.items())
    random.shuffle(graph1_combinations)
    for i in range(num_comparisons):
        G1 = graph1_combinations[i][1]
        G2 = graph1_combinations[i + 1][1]
        similarity = get_graphlet_kernel(G1, G2, k)
        same_class_similarities1.append(similarity)

    
    # Different class comparisons (graph0 vs graph1)
    graph0_combinations = list(graphs_0.items())
    graph1_combinations = list(graphs_1.items())
    random.shuffle(graph0_combinations)
    random.shuffle(graph1_combinations)
    for i in range(num_comparisons):
        G1 = graph0_combinations[i][1]
        G2 = graph1_combinations[i][1]
        similarity = get_graphlet_kernel(G1, G2, k)
        diff_class_similarities.append(similarity)
    
    # Calculate statistics
    def calculate_stats(similarities):
        mean = np.mean(similarities)
        std = np.std(similarities)
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        return mean, std, min_sim, max_sim
    
    same_class_stats0 = calculate_stats(same_class_similarities0)
    same_class_stats1 = calculate_stats(same_class_similarities1)
    diff_class_stats = calculate_stats(diff_class_similarities)
    
    return same_class_stats0, same_class_stats1, diff_class_stats

# Function to generate data
def generate_comparison_data(graphs_0, graphs_1, k_values):
    table_data = {}

    for k in k_values:
        same_class_stats0, same_class_stats1, diff_class_stats = calculate_similarity_for_k(graphs_0, graphs_1, k)
        table_data[k] = [same_class_stats0, same_class_stats1, diff_class_stats]
    
    return table_data

if __name__ == "__main__":
    # Create the MOLT-4 networkx graphs
    adj_list = get_adj_list('MOLT-4/MOLT-4_A.txt')
    graph_indicators = get_graph_data('MOLT-4/MOLT-4_graph_indicator.txt')
    graph_set = set(graph_indicators)
    node_labels = get_graph_data('MOLT-4/MOLT-4_node_labels.txt')
    edge_labels = get_graph_data('MOLT-4/MOLT-4_edge_labels.txt')
    graph_labels = get_graph_data('MOLT-4/MOLT-4_graph_labels.txt')
    graphs = create_graphs_by_label(adj_list, graph_indicators, node_labels, edge_labels, graph_labels)

    # Get runtime data
    runtimes0, runtimes1, runtimes_combined = get_runtimes(graphs, 6)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    k_values = [1, 2, 3, 4, 5]
    
    # Plot for graphs[0], graphs[1], and combined
    plt.plot(k_values, runtimes0, label='Graphs[0] Average Runtime', marker='o')
    plt.plot(k_values, runtimes1, label='Graphs[1] Average Runtime', marker='x')
    plt.plot(k_values, runtimes_combined, label='Combined Average Runtime', marker='s')

    # Adding labels and title
    plt.xlabel("k Value")
    plt.ylabel("Average Runtime (seconds)")
    plt.title("Average Runtime vs. k Value")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Get comparison data
    k_values = [1, 2, 3, 4]
    comparison_data = generate_comparison_data(graphs[0], graphs[1], k_values)
    print(comparison_data)
