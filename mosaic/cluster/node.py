import time
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from benchmarking.benchmark import MOSAICSimScheduler, SimTask, INTERFERENCE_MATRIX

class EdgeNode:
    """
    Abstracts a single edge node running an independent instance of MOSAIC.
    """
    def __init__(self, node_id: str, scheduler_class=MOSAICSimScheduler):
        self.node_id = node_id
        # Independent scheduler and interference matrix per node
        self.scheduler = scheduler_class()
        
        # Telemetry
        self.migrations_in = 0
        self.migrations_out = 0

    def admit_task(self, task: SimTask) -> str:
        """Attempts to admit a task locally."""
        return self.scheduler.admit(task)
        
    def complete_task(self, task_id: str, actual_ms: float):
        self.scheduler.complete(task_id, actual_ms)

    def get_cpu_utilization(self) -> float:
        """Percentage of MAX_RUN slots occupied."""
        return len(self.scheduler._running) / max(1, self.scheduler.MAX_RUN)

    def get_pmu_pressure(self) -> float:
        """Aggregated interference score of running tasks."""
        rl = list(self.scheduler._running.values())
        rc = [t.cls for t in rl]
        
        pressure = 0.0
        for task in rl:
            for rcls in rc:
                if task.cls != rcls and task.cls in INTERFERENCE_MATRIX and rcls in INTERFERENCE_MATRIX[task.cls]:
                    ipc_deg, lat = INTERFERENCE_MATRIX[task.cls][rcls]
                    pressure += ipc_deg
        return pressure
        
    def get_llc_pressure(self) -> float:
        """Synthetic approximation of LLC thrashing based on running classes."""
        # E.g. analytics_batch and model_update drive heavy LLC pressure
        heavy_classes = ["analytics_batch", "model_update", "cache_thrash", "retry_storm"]
        rl = list(self.scheduler._running.values())
        return sum(1.0 for t in rl if t.cls in heavy_classes) / max(1, self.scheduler.MAX_RUN)
        
    def check_migration_need(self) -> list[SimTask]:
        """
        Identifies tasks that should be migrated due to extreme local overload
        or LLC pressure spikes.
        """
        candidates = []
        now = time.monotonic() * 1000
        
        # If queue is backing up, nominate youngest tasks for migration
        if len(self.scheduler._queue) > 5 or self.get_pmu_pressure() > 1.5:
            # Pick tasks that haven't started running yet
            for task in self.scheduler._queue[-2:]:
                candidates.append(task)
                
        return candidates
        
    def remove_queued_task(self, task_id: str) -> SimTask:
        for i, t in enumerate(self.scheduler._queue):
            if t.task_id == task_id:
                self.migrations_out += 1
                return self.scheduler._queue.pop(i)
        return None
