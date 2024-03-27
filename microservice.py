#MICROSERVICE  
class Microservice:
    def __init__(self, service_id, name, cpu_requirement, memory_requirement, bandwidth_requirement, latency_threshold, server=None):
        self.service_id = service_id
        self.name = name
        self.cpu_requirement = cpu_requirement
        self.memory_requirement = memory_requirement
        self.bandwidth_requirement = bandwidth_requirement
        self.latency_threshold = latency_threshold
        self.server = server  # HOST SERVER

    def __str__(self):
        server_id = self.server.server_id if self.server else "Not deployed"
        return (f"Microservice(name={self.name}, CPU={self.cpu_requirement}, "
                f"Memory={self.memory_requirement}, Bandwidth={self.bandwidth_requirement}, "
                f"Latency Threshold={self.latency_threshold}ms, Server={server_id})")

def adjust_microservice_requirements(microservices, client_clusters):
    # Initialize or reset microservice adjustments
    for ms in microservices:
        ms.adjusted_cpu_requirement = ms.cpu_requirement
        ms.adjusted_memory_requirement = ms.memory_requirement
        ms.adjusted_bandwidth_requirement = ms.bandwidth_requirement
        ms.min_latency_requirement = float('inf')

    # Adjust based on aggregated client demands and latency requirements
    for cluster in client_clusters:
        for service_id in cluster.microservices:
            # Find the corresponding microservice
            microservice = next((ms for ms in microservices if ms.service_id == service_id), None)
            if microservice:
                # Example adjustment logic
                microservice.adjusted_cpu_requirement += cluster.total_demand * cpu_demand_factor
                microservice.adjusted_memory_requirement += cluster.total_demand * memory_demand_factor
                microservice.adjusted_bandwidth_requirement += cluster.total_demand * bandwidth_demand_factor
                microservice.min_latency_requirement = min(microservice.min_latency_requirement, cluster.latency_requirement)

