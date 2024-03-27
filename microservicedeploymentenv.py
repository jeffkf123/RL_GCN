
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from collections import namedtuple, deque
import random
import numpy as np


class MicroserviceDeploymentEnv:
        
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    def __init__(self, graph, microservices):
        self.graph = graph  
        self.original_microservices = microservices
        self.microservices = None
        self.current_microservice_index = -1

        self.current_microservice = None
        self.state_size = None
        self.action_size = len(graph.nodes)  # Dynamically determined by the number of servers in the graph

    def reset(self):
            # Reset the environment to its initial state
            self.microservices = list(self.original_microservices)  # Create a fresh copy of the microservices for the new episode
            self.current_microservice_index = 0  # Reset to the first microservice
            self.current_microservice = self.microservices[self.current_microservice_index]
            
            # Reset servers to their original state
            for server in self.graph.nodes.values():
                server.available_cpu = server.total_cpu
                server.available_memory = server.total_memory
                server.available_bandwidth = server.total_bandwidth
                server.hosted_services.clear()
            
            # Optionally, reset any other environment state variables
            ...
            
            return self.get_state()  # Return the initial state of the environment

    @staticmethod        
    def  normalize(value, min_value, max_value):
        """Normalize a value to 0-1 range for optimal dqn input"""

        return (value - min_value) / (max_value - min_value) if max_value > min_value else 0
    

    def get_state(self):
        # Node features
        x = []  # List to hold node features
        edge_index = [[], []]  # List to hold source and target nodes of each edge
        edge_attr = []  # List to hold edge attributes (optional)

        # Prepare node features (e.g., normalized CPU, memory, bandwidth)
        for server_id, server in self.graph.nodes.items():
            normalized_cpu = self.normalize(server.available_cpu, 0, server.total_cpu)
            normalized_memory = self.normalize(server.available_memory, 0, server.total_memory)
            normalized_bandwidth = self.normalize(server.available_bandwidth, 0, server.total_bandwidth)
            server_features = [normalized_cpu, normalized_memory, normalized_bandwidth]
            x.append(server_features)

        # Prepare edge connections and (optional) edge attributes
        for source_id, connections in self.graph.edges.items():
            for target_id, connection_info in connections:
                source_index = list(self.graph.nodes.keys()).index(source_id)
                target_index = list(self.graph.nodes.keys()).index(target_id)
                edge_index[0].append(source_index)
                edge_index[1].append(target_index)
                # Optional: Prepare edge attributes, e.g., latency
                edge_attr.append([connection_info['latency'], connection_info['bandwidth']])
        
        # Convert lists to tensors
        x_tensor = torch.tensor(x, dtype=torch.float)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)

        # Create the PyTorch Geometric Data object
        data = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)

        return data

    def execute_action(self, action):
        # Check if there are any microservices left to deploy
        if self.current_microservice_index >= len(self.microservices):
            # No microservices left, end the episode
            return None, 0, True  # Assuming no state, no reward, and done is True

        # Action is the index corresponding to a server in the graph.nodes dictionary
        server_ids = list(self.graph.nodes.keys())
        try:
            selected_server_id = server_ids[action]
            selected_server = self.graph.nodes[selected_server_id]
            # Deploy the microservice to the selected server
            if selected_server.meets_requirements(
                self.current_microservice.cpu_requirement,
                self.current_microservice.memory_requirement,
                self.current_microservice.bandwidth_requirement
            ):
                success = selected_server.deploy_microservice(self.current_microservice)
                reward = self.calculate_reward(action, success)  # Calculate reward based on action success
                
                # Move to the next microservice, update state
                self.current_microservice_index += 1
                if self.current_microservice_index < len(self.microservices):
                    self.current_microservice = self.microservices[self.current_microservice_index]
                    next_state = self.get_state()  # Get the new state after deployment
                else:
                    # No more microservices to deploy, end of episode
                    next_state = None
                    self.current_microservice = None
                
                # Check if the episode should end
                done = self.current_microservice_index >= len(self.microservices)
                
                return next_state, reward, done
            else:
                # Deployment failed due to insufficient resources try another server
                # or to move to the next microservice
                #penalize and move to the next microservice
                self.current_microservice_index += 1
                if self.current_microservice_index < len(self.microservices):
                    self.current_microservice = self.microservices[self.current_microservice_index]
                    next_state = self.get_state()
                else:
                    next_state = None
                    self.current_microservice = None
                return next_state, -1, self.current_microservice_index >= len(self.microservices)  # Penalize and potentially end episode
        except IndexError:
            # Handle invalid action (e.g., action index out of range)
            return None, -1, False  # Return a default failed state with a penalty but not ending the episode

    
    def calculate_reward(self, action, success):
        """
        Calculate the reward for deploying a microservice based on the action's success and system state.
        
        Parameters:
            action (int): The destination server of the action attempted
            success: Whether the microservice deployment was successful.
        
        Returns the calculated reward for the action.
             
        """
        if not self.current_microservice:
            # No microservice is selected for deployment...system error.
            return -10  # Penalize heavily
        
        server_ids = list(self.graph.nodes.keys())
        try:
            selected_server_id = server_ids[action]
            selected_server = self.graph.nodes[selected_server_id]
        except IndexError:
            # Action led to an invalid server selection
            return -5
        
        if not success:
            # Deployment failed. Penalize to a degree but less than system errors.
            return -2

        # If deployment was successful, positive reward is calculated based on several factors, we can supplement it if necessary
        reward = 0
        
        # Factor 1: Resource Utilization Efficiency. Encourage efficient use of server resources without overloading.

        cpu_utilization = (selected_server.total_cpu - selected_server.available_cpu) / selected_server.total_cpu
        memory_utilization = (selected_server.total_memory - selected_server.available_memory) / selected_server.total_memory
        bandwidth_utilization = (selected_server.total_bandwidth - selected_server.available_bandwidth) / selected_server.total_bandwidth
        
        # Average utilization leads to balanced use of resources.
        avg_utilization = (cpu_utilization + memory_utilization + bandwidth_utilization) / 3
        reward += 5 * avg_utilization  
        
        # Factor 2: Network Performance. Penalize if the deployment significantly impacts network latency or does not meet latency requirements.

        if selected_server.calculate_latency_priority(self.current_microservice) > self.current_microservice.latency_threshold:
            # Penalize for exceeding latency threshold, scaled by how much it was exceeded.
            reward -= 5 * (selected_server.calculate_latency_priority(self.current_microservice) / self.current_microservice.latency_threshold)
        
        # Factor 3: Load Balancing. Reward deployments that help balance the load across the network.
        
        
        load_imbalance = self.graph.calculate_load_imbalance()  # Assess load balance across servers.
        avg_imbalance = (load_imbalance['cpu_std'] + load_imbalance['memory_std'] + load_imbalance['bandwidth_std']) / 3

        reward -= 5 * avg_imbalance  # Penalize based on degree of imbalance to encourage load balancing.
        
        # OTHERS
        
        return reward
