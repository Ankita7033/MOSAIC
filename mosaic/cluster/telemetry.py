import time
from typing import Dict, Any

class ClusterTelemetry:
    """
    Aggregates metrics from all nodes to enable PMU heatmaps, 
    overload visualization, and interference propagation tracking.
    """
    def __init__(self):
        self.snapshots = []
        
    def aggregate(self, nodes: Dict[str, Any]) -> Dict[str, dict]:
        """Collects a point-in-time snapshot of the cluster state."""
        snapshot = {}
        for node_id, node in nodes.items():
            snapshot[node_id] = {
                "cpu_util": node.get_cpu_utilization(),
                "queue_depth": len(node.scheduler._queue),
                "active_tasks": len(node.scheduler._running),
                "pmu_pressure": node.get_pmu_pressure(),
                "llc_pressure": node.get_llc_pressure(),
                "rejections": node.scheduler._rejected,
                "migrations_out": node.migrations_out,
                "migrations_in": node.migrations_in
            }
            
        self.snapshots.append({
            "ts": time.monotonic(),
            "nodes": snapshot
        })
        return snapshot

    def export_latest(self) -> dict:
        if not self.snapshots:
            return {}
        return self.snapshots[-1]
