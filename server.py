class Server:
    def __init__(self, server_id, ip_address, total_cpu, total_memory, total_bandwidth, network_distance, hosted_services=None):
        self.server_id = server_id
        self.network_distance = network_distance
        self.ip_address = ip_address
        self.total_cpu = total_cpu
        self.total_memory = total_memory
        self.total_bandwidth = total_bandwidth
        self.available_cpu = total_cpu  # Initially, all resources are available
        self.available_memory = total_memory
        self.available_bandwidth = total_bandwidth
        self.hosted_services = hosted_services if hosted_services is not None else []

    def calculate_agg_resources(self): # not planning to use it now
        """Calculate the average resource utilization as an aggregate metric."""
        return (self.available_cpu + self.available_memory + self.available_bandwidth) / 3

    
    def add_service_function(self, service_function):
        self.hosted_services.append(service_function)

    def remove_service_function(self, service_function):
        self.hosted_services.remove(service_function)

    def list_service_functions(self):
        return self.hosted_services
    
    def calculate_latency_priority(self, microservice):
        """Simulate latency based on network distance and server load, in real life we would have the microservice ping the server."""
        base_latency = self.network_distance * 0.5  # Simplified calculation
        return base_latency
    
    def meets_requirements(self, cpu_requirement, memory_requirement, bandwidth_requirement):
        """Check if the server has enough resources to host the microservice."""
        return (self.available_cpu >= cpu_requirement and
                self.available_memory >= memory_requirement and
                self.available_bandwidth >= bandwidth_requirement)
    def deploy_microservice(self, microservice):
        #Deploy service and subtract required resources
        if not self.meets_requirements(microservice.cpu_requirement, microservice.memory_requirement, microservice.bandwidth_requirement):
            return False  # Deployment fails due to insufficient resources
        
        self.available_cpu -= microservice.cpu_requirement
        self.available_memory -= microservice.memory_requirement
        self.available_bandwidth -= microservice.bandwidth_requirement
        self.hosted_services.append(microservice.service_id)
        microservice.server=self
        return True

    def remove_microservice(self, microservice):
        #remove a service and re-add used resources to server
        if microservice.service_id in self.hosted_services:
            self.available_cpu += microservice.cpu_requirement
            self.available_memory += microservice.memory_requirement
            self.available_bandwidth += microservice.bandwidth_requirement
            self.hosted_services.remove(microservice.service_id)
            microservice.server= None

    def __str__(self):
        return (f"Server(ID={self.server_id}, IP={self.ip_address}, "
                f"CPU={self.available_cpu}, Memory={self.available_memory}, "
                f"Bandwidth={self.available_bandwidth}, Distance={self.network_distance}, "
                f"Hosted_Services={len(self.hosted_services)})")
