import random

class MigrationEngine:
    """
    Models the penalty of moving tasks between edge nodes.
    Includes network transfer delays and cache warmup penalties.
    """
    def __init__(self, network_model):
        self.network = network_model
        
    def calculate_cost(self, task) -> float:
        """Returns the migration penalty in milliseconds."""
        # Base network serialization and transfer delay
        network_ms = self.network.get_rpc_delay_ms() * 2.0
        
        # Cache warmup penalty (larger tasks = more penalty)
        warmup_ms = getattr(task, 'mem_mb', 256) * 0.05
        
        jitter = random.uniform(0.9, 1.1)
        return (network_ms + warmup_ms) * jitter

    def execute_migration(self, task, source_node, target_node) -> bool:
        """
        Simulates the actual migration. 
        In this simulation, we just transfer the object and add the penalty to its service time.
        """
        penalty_ms = self.calculate_cost(task)
        
        # Detach from source
        pulled_task = source_node.remove_queued_task(task.task_id)
        if not pulled_task:
            return False
            
        # Add penalty to service time to simulate the cost on the new node
        pulled_task.service_ms += penalty_ms
        target_node.migrations_in += 1
        
        # Submit to target
        res = target_node.admit_task(pulled_task)
        return res in ("admitted", "force_admitted", "queued")
