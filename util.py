
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
#SERVER
from server import Server
from client import Client
from client import ClientCluster
from microservice import Microservice, adjust_microservice_requirements
from networkgraph import NetworkGraph

# GRAPH
import numpy as np
import heapq
import networkx as nx
import matplotlib.pyplot as plt

#ENVIRONMENT
from microservicedeploymentenv import MicroserviceDeploymentEnv

from dqnagent import GCN, DQN_GCN, ReplayMemory, Transition, DQN_Agent

def visualize_deployment(client_clusters, servers, microservices):
    G = nx.Graph()

    # Add client cluster nodes
    for cluster in client_clusters:
        G.add_node(f"Cluster {cluster.region}", type='cluster', demand=cluster.total_demand)

    # Add server nodes
    for server in servers:
        G.add_node(server.server_id, type='server', capacity=server.total_cpu)  # Simplified to CPU capacity for demonstration

    # Add edges based on microservice deployments
    for ms in microservices:
        if ms.server:  # If the microservice is deployed
            for cluster in client_clusters:
                if ms.service_id in cluster.microservices:
                    G.add_edge(f"Cluster {cluster.region}", ms.server.server_id, microservice=ms.service_id)

    # Visualization
    pos = nx.spring_layout(G)  # Position nodes using the spring layout
    # Draw clusters
    cluster_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'cluster']
    nx.draw_networkx_nodes(G, pos, nodelist=cluster_nodes, node_color='lightblue', label='Client Cluster')
    # Draw servers
    server_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'server']
    nx.draw_networkx_nodes(G, pos, nodelist=server_nodes, node_color='lightgreen', label='Server')
    # Draw edges
    nx.draw_networkx_edges(G, pos)
    # Labels
    nx.draw_networkx_labels(G, pos)
    # Legend
    plt.legend(loc="upper left")

    plt.axis('off')
    plt.show()



def visualize_load_imbalances(load_imbalances):
    """
    Visualizes load imbalances over episodes using matplotlib.

    Parameters:
        load_imbalances (list): List of load imbalances over episodes.
    """
    print(load_imbalances)
    plt.figure(figsize=(10, 6))
    plt.plot(load_imbalances, label='Load Imbalance')
    plt.xlabel('Episode')
    plt.ylabel('Load Imbalance')
    plt.title('Load Imbalance Over Episodes')
    plt.legend()
    plt.show()


def visualize_with_clusters(network_graph, client_clusters):
    G = nx.Graph()

    # Add server nodes
    for server_id in network_graph.nodes:
        G.add_node(server_id, type='server', layer=1)

    # Add client cluster nodes
    for idx, cluster in enumerate(client_clusters, start=1):
        G.add_node(f"Cluster {idx}", type='client_cluster', layer=0)

    # Connect servers with each other based on existing connections
    for server_id, connections in network_graph.edges.items():
        for target_id, info in connections:
            G.add_edge(server_id, target_id, weight=info['latency'])

    # Connect client clusters to servers based on the deployment of their microservices
    for idx, cluster in enumerate(client_clusters, start=1):
        for service_id in cluster.microservices:
            # Find the server where this service is deployed
            for server in network_graph.nodes.values():
                if service_id in server.hosted_services:
                    G.add_edge(f"Cluster {idx}", server.server_id)
                    break

    pos = nx.multipartite_layout(G, subset_key="layer")
    nx.draw(G, pos, with_labels=True, node_size=700)
    plt.show()



def list_servers_and_microservices(graph):
    print("\nListing all servers and deployed microservices:")
    for server_id, server in graph.nodes.items():
        deployed_services = [ms_id for ms_id in server.hosted_services]
        print(f"Server {server_id} ({server.ip_address}): Deployed Microservices -> {deployed_services}")

def plot_resource_utilization_for_all_servers(graph):
    for server_id, server in graph.nodes.items():
        utilized_cpu = server.total_cpu - server.available_cpu
        utilized_memory = server.total_memory - server.available_memory
        utilized_bandwidth = server.total_bandwidth - server.available_bandwidth

        resources = ['CPU', 'Memory', 'Bandwidth']
        total_resources = [server.total_cpu, server.total_memory, server.total_bandwidth]
        utilized_resources = [utilized_cpu, utilized_memory, utilized_bandwidth]

        fig, ax = plt.subplots()
        ax.bar(resources, total_resources, label='Total')
        ax.bar(resources, utilized_resources, label='Utilized', width=0.5)

        ax.set_ylabel('Resources')
        ax.set_title(f'Server {server_id} Resource Utilization')
        ax.legend()

        plt.show()

def plot_resource_utilization_for_all_servers_pct(graph):
    for server_id, server in graph.nodes.items():
        utilized_cpu_percent = ((server.total_cpu - server.available_cpu) / server.total_cpu) * 100 if server.total_cpu else 0
        utilized_memory_percent = ((server.total_memory - server.available_memory) / server.total_memory) * 100 if server.total_memory else 0
        utilized_bandwidth_percent = ((server.total_bandwidth - server.available_bandwidth) / server.total_bandwidth) * 100 if server.total_bandwidth else 0

        resources = ['CPU', 'Memory', 'Bandwidth']
        utilized_resources_percent = [utilized_cpu_percent, utilized_memory_percent, utilized_bandwidth_percent]

        fig, ax = plt.subplots()
        ax.bar(resources, utilized_resources_percent, color=['blue', 'orange', 'green'])

        ax.set_ylabel('Utilization (%)')
        ax.set_title(f'Server {server_id} Resource Utilization')
        ax.set_ylim(0, 100) 

       
        for i, utilization in enumerate(utilized_resources_percent):
            ax.text(i, utilization + 2, f'{utilization:.2f}%', ha='center', va='bottom')

        plt.show()


