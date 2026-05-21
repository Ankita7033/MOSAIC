import time
import random

class NetworkModel:
    """
    Simulates network latency and bandwidth constraints for cluster communications.
    Provides realistic RTT delays for placement, telemetry, and migration.
    """
    def __init__(self, base_rtt_ms: float = 3.0, jitter_ms: float = 1.0):
        self.base_rtt_ms = base_rtt_ms
        self.jitter_ms = jitter_ms
        
    def get_telemetry_delay_ms(self) -> float:
        """Simulate delay in receiving telemetry from a node."""
        return max(0.5, (self.base_rtt_ms / 2) + random.uniform(-self.jitter_ms, self.jitter_ms))
        
    def get_rpc_delay_ms(self) -> float:
        """Simulate a placement decision RPC round-trip."""
        return max(1.0, self.base_rtt_ms + random.uniform(-self.jitter_ms, self.jitter_ms))
        
    def simulate_delay(self, ms: float):
        """Actually sleep the thread to simulate blocking network calls."""
        if ms > 0:
            time.sleep(ms / 1000.0)
