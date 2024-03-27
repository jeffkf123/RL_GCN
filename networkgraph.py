import numpy as np
import heapq
import networkx as nx
import matplotlib.pyplot as plt
class NetworkGraph:
    def __init__(self):
        self.nodes = {}  # Stores server objects with server_id as key
        self.edges = {}  # Stores connections and their properties

    def add_server(self, server):
        """Add a server"""
        self.nodes[server.server_id] = server
        self.edges[server.server_id] = []

    def connect_servers(self, server_id1, server_id2, latency, bandwidth):
        """Connect two servers with specified latency and bandwidth."""
        if server_id1 not in self.nodes or server_id2 not in self.nodes:
            raise ValueError("One or both of the servers not found in the graph.")
        
        connection_info = {'latency': latency, 'bandwidth': bandwidth}
        self.edges[server_id1].append((server_id2, connection_info))
        self.edges[server_id2].append((server_id1, connection_info))

    def get_server_connections(self, server_id):
        """Retrieve a server's connections and their properties."""
        return self.edges.get(server_id, [])

    def __str__(self):
        """Provide a string overload of the graph for debugging purposes."""
        description = "Network Graph:\n"
        for server_id in self.nodes:
            description += f"Server {server_id} connections: {self.edges[server_id]}\n"
        return description
    
    def find_shortest_path(self, start_server_id, end_server_id):
        """Find the shortest path from start to end server based on latency using Dijkstra"""
        distances = {server_id: float('inf') for server_id in self.nodes}
        distances[start_server_id] = 0
        priority_queue = [(0, start_server_id)]
        predecessor = {server_id: None for server_id in self.nodes}

        while priority_queue:
            current_distance, current_server_id = heapq.heappop(priority_queue)
            if current_server_id == end_server_id:
                break

            for neighbor, connection_info in self.get_server_connections(current_server_id):
                distance = current_distance + connection_info['latency']
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessor[neighbor] = current_server_id
                    heapq.heappush(priority_queue, (distance, neighbor))

        # Reconstructing path
        path = []
        current = end_server_id
        while current is not None:
            path.append(current)
            current = predecessor[current]
        path.reverse()

        return path if path[0] == start_server_id else None
    
    def average_network_latency(self):
        """Calculate the average latency across all connections in the graph."""
        total_latency, count = 0, 0
        for server_id in self.edges:
            for connection in self.edges[server_id]:
                _, connection_info = connection
                total_latency += connection_info['latency']
                count += 1
        return total_latency / count if count > 0 else 0
    

    def update_connection(self, server_id1, server_id2, latency=None, bandwidth=None):
        """Update the properties of a connection between two servers."""
        if server_id1 in self.edges and server_id2 in self.edges:
            for i, (target_id, info) in enumerate(self.edges[server_id1]):
                if target_id == server_id2:
                    if latency is not None:
                        self.edges[server_id1][i][1]['latency'] = latency
                    if bandwidth is not None:
                        self.edges[server_id1][i][1]['bandwidth'] = bandwidth
                    break
            # Bidirectional repeat for the reverse connection
            for i, (target_id, info) in enumerate(self.edges[server_id2]):
                if target_id == server_id1:
                    if latency is not None:
                        self.edges[server_id2][i][1]['latency'] = latency
                    if bandwidth is not None:
                        self.edges[server_id2][i][1]['bandwidth'] = bandwidth
                    break
        else:
            raise ValueError("One or both of the servers not found in the graph.")
        

    def calculate_load_imbalance(self):
        """
        Calculate the imbalance in server load across the network.
        
        Returns a dictionary with the standard deviation of CPU, memory, and bandwidth utilizations,
            indicating the level of imbalance. Lower values indicate a more balanced load.
        """
        cpu_utilizations = []
        memory_utilizations = []
        bandwidth_utilizations = []
        
        for server_id, server in self.nodes.items():
            cpu_util = (server.total_cpu - server.available_cpu) / server.total_cpu if server.total_cpu else 0
            memory_util = (server.total_memory - server.available_memory) / server.total_memory if server.total_memory else 0
            bandwidth_util = (server.total_bandwidth - server.available_bandwidth) / server.total_bandwidth if server.total_bandwidth else 0
            
            cpu_utilizations.append(cpu_util)
            memory_utilizations.append(memory_util)
            bandwidth_utilizations.append(bandwidth_util)
        
        # Calculate standard deviation for each resource utilization
        cpu_std = np.std(cpu_utilizations) if cpu_utilizations else 0
        memory_std = np.std(memory_utilizations) if memory_utilizations else 0
        bandwidth_std = np.std(bandwidth_utilizations) if bandwidth_utilizations else 0
        
        return {
            'cpu_std': cpu_std,
            'memory_std': memory_std,
            'bandwidth_std': bandwidth_std
        }
    def get_average_latency_to_other_servers(self, server_id):
        """Calculate the average latency from a specified server to all other servers."""
        if server_id not in self.edges:
            raise ValueError(f"Server {server_id} not found in the graph.")
        
        total_latency = 0
        connections = 0
        for connection in self.edges[server_id]:
            neighbor_id, connection_info = connection
            total_latency += connection_info['latency']
            connections += 1
        
        # We don't divide by zero if a server has no connections
        if connections == 0:
            return float('inf')  
        
        return total_latency / connections
    
    def visualize(self):
        """Visualize the network graph."""
        G = nx.Graph()
        for server_id in self.nodes:
            G.add_node(server_id)
        for server_id, connections in self.edges.items():
            for target_id, info in connections:
                G.add_edge(server_id, target_id, weight=info['latency'])
        
        pos = nx.spring_layout(G)  # positions for all nodes
        nx.draw_networkx_nodes(G, pos, node_size=700)
        nx.draw_networkx_edges(G, pos, width=2)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
        
        edge_labels = dict([((u, v,), f"{d['weight']}ms")
                            for u, v, d in G.edges(data=True)])
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.axis('off')
        plt.show()
