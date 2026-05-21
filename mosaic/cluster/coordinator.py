import time
from typing import List, Dict
from cluster.node import EdgeNode
from cluster.network import NetworkModel
from cluster.migration import MigrationEngine
from cluster.telemetry import ClusterTelemetry

class CentralCoordinator:
    """
    Global Placement and Overload Coordinator for the MOSAIC Edge Cluster.
    Uses a centralized architecture to assign tasks based on PMU pressure and queue depth.
    """
    def __init__(self, nodes: List[EdgeNode]):
        self.nodes: Dict[str, EdgeNode] = {n.node_id: n for n in nodes}
        self.network = NetworkModel(base_rtt_ms=2.0)
        self.migration = MigrationEngine(self.network)
        self.telemetry = ClusterTelemetry()
        
        # Placement Weights
        self.w_cpu = 1.0
        self.w_queue = 1.5
        self.w_pmu = 2.0
        self.w_llc = 2.5

    def collect_telemetry(self):
        """Simulates periodic telemetry gathering from all nodes."""
        delay = self.network.get_telemetry_delay_ms()
        self.network.simulate_delay(delay)
        return self.telemetry.aggregate(self.nodes)

    def _score_node(self, node: EdgeNode) -> float:
        """
        Calculates the placement penalty score. Lower is better.
        Score = w1*CPU + w2*Queue + w3*PMU + w4*Interference
        """
        cpu_util = node.get_cpu_utilization()
        queue_depth = len(node.scheduler._queue)
        pmu_pressure = node.get_pmu_pressure()
        llc_pressure = node.get_llc_pressure()
        
        return (self.w_cpu * cpu_util) + \
               (self.w_queue * queue_depth) + \
               (self.w_pmu * pmu_pressure) + \
               (self.w_llc * llc_pressure)

    def dispatch_task(self, task) -> str:
        """Finds the best node and dispatches the task."""
        # 1. Simulate RPC routing delay
        self.network.simulate_delay(self.network.get_rpc_delay_ms())
        
        # 2. Score all nodes
        best_node = None
        min_score = float('inf')
        
        for n_id, node in self.nodes.items():
            score = self._score_node(node)
            if score < min_score:
                min_score = score
                best_node = node
                
        # 3. Admit to the chosen node
        if best_node:
            task.assigned_node = best_node.node_id
            return best_node.admit_task(task)
        return "rejected"

    def reconcile_overload(self):
        """
        Checks for nodes experiencing severe localized bursts and migrates
        tasks to less loaded nodes.
        """
        for n_id, node in self.nodes.items():
            candidates = node.check_migration_need()
            for task in candidates:
                # Find a better home
                current_score = self._score_node(node)
                
                best_target = None
                best_improvement = 0
                
                for t_id, target in self.nodes.items():
                    if t_id == n_id: continue
                    target_score = self._score_node(target)
                    improvement = current_score - target_score
                    
                    if improvement > best_improvement and improvement > 1.0: # threshold to prevent thrashing
                        best_improvement = improvement
                        best_target = target
                        
                if best_target:
                    # Execute migration
                    success = self.migration.execute_migration(task, node, best_target)
                    if success:
                        task.assigned_node = best_target.node_id
