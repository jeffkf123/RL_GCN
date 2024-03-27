
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

#UTILITIES
from util import visualize_deployment
from util import visualize_with_clusters
from util import list_servers_and_microservices
from util import plot_resource_utilization_for_all_servers
from util import plot_resource_utilization_for_all_servers_pct
from util import visualize_load_imbalances


print(torch.cuda.is_available())
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


network_graph = NetworkGraph()

# Cluster A Servers
cluster_a_servers = ["serverA1", "serverA2", "serverA3"]
for server_id in cluster_a_servers:
    network_graph.add_server(Server(server_id, "192.168.1.x", 200, 300, 2000, 5))

# Cluster B Servers
cluster_b_servers = ["serverB1", "serverB2", "serverB3"]
for server_id in cluster_b_servers:
    network_graph.add_server(Server(server_id, "192.168.2.x", 200, 300, 2000, 5))

# Bottleneck Link - clusters with higher latency / limited bandwidth
bottleneck_link_latency = 50  # Higher latency
bottleneck_link_bandwidth = 500  # Limited bandwidth
for server_a in cluster_a_servers:
    for server_b in cluster_b_servers:
        network_graph.connect_servers(server_a, server_b, bottleneck_link_latency, bottleneck_link_bandwidth)

microservices = [
    Microservice("service1", "Auth", 50, 100, 500, 25),
    Microservice("service2", "Database", 100, 150, 1000, 30),
    Microservice("service3", "Cache", 75, 125, 750, 20),
    Microservice("service4", "Logging", 40, 90, 400, 18),
    Microservice("service5", "Payment", 60, 110, 600, 22),
    Microservice("service6", "Search", 85, 135, 850, 26),

]
client_clusters = [
    ClientCluster(region=1, microservices=["service1", "service3"], total_demand=300, latency_requirement=100),
    ClientCluster(region=2, microservices=["service2", "service4", "service5"], total_demand=500, latency_requirement=150)
]

env = MicroserviceDeploymentEnv(network_graph, microservices)
agent = DQN_Agent(num_features=3, num_actions=len(cluster_a_servers) + len(cluster_b_servers), device=device)

network_graph.visualize()

load_imbalances = []  # List to store load imbalance of each episode


num_episodes = 50  # Number of episodes to run for testing
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        action_scalar = action.squeeze().item()
        next_state, reward, done = env.execute_action(action_scalar)
        total_reward += reward

        agent.memory.push(state, action, next_state, torch.tensor([reward], dtype=torch.float), done)
        state = next_state
        agent.optimize_model(batch_size=32)

    if episode % 10 == 0:
        agent.update_target_net()

    load_imbalance = network_graph.calculate_load_imbalance()
    avg_imbalance = (load_imbalance['cpu_std'] + load_imbalance['memory_std'] + load_imbalance['bandwidth_std']) / 3        

    load_imbalances.append(avg_imbalance)
    print(f'Episode {episode+1}, Total Reward: {total_reward}, Imbalance: {avg_imbalance}')
    list_servers_and_microservices(network_graph)


        
network_graph.visualize()
visualize_load_imbalances(load_imbalances)
list_servers_and_microservices(network_graph)
plot_resource_utilization_for_all_servers_pct(network_graph)
plot_resource_utilization_for_all_servers(network_graph)
plot_resource_utilization_for_all_servers_pct(network_graph)

# List servers and their deployed microservices in the new environment
list_servers_and_microservices(network_graph)
# Assuming 'agent.policy_net' is your model
visualize_with_clusters(network_graph, client_clusters)
torch.save(agent.policy_net.state_dict(), 'dqn_model.pth')
