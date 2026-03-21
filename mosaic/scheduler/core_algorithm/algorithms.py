"""
MOSAIC Core Scheduling Algorithms
===================================
Contains the mathematical core of MOSAIC:

1. Urgency Score -- dynamic, deadline-driven priority
2. Interference Admission Check -- pairwise compatibility
3. Fairness Monitor -- Jain's Fairness Index tracking
4. Starvation Guard -- prevents background task starvation
5. Energy-aware DVFS trigger logic
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from workload_taxonomy import WORKLOAD_CLASSES, TIER_WEIGHTS, INTERFERENCE_MATRIX

# -- Constants ------------------------------------------------------------------
INTERFERENCE_IPC_THRESHOLD  = 0.35   # block colocation if ipc_deg > 35%
DEADLINE_BUDGET_FRACTION    = 0.40   # don't let interference consume >40% of budget
STARVATION_TIMEOUT_MULT     = 3.0    # task at 3× its deadline age = starvation
MAX_CONCURRENT_TASKS        = 8
DVFS_POWER_THRESHOLD        = 0.88   # throttle at 88% of power cap
CONFIDENCE_SAFETY_FACTOR    = 0.50   # extra margin for low-confidence entries


# -- Urgency Score -------------------------------------------------------------

def compute_urgency(deadline_ms: int, elapsed_ms: float,
                    tier: int, priority: int = 2) -> float:
    """
    MOSAIC Urgency Score formula:

        U(t) = tier_weight × priority_weight
               -----------------------------------------
               max(ε, deadline_remaining_ms / deadline_ms)

    Properties:
    - Monotonically increasing as deadline approaches
    - Tier 1 (CRITICAL) always dominates tier 2+
    - Past-deadline tasks → U = ∞ (guaranteed immediate scheduling)
    - Smooth, continuous, no hard priority inversion

    Args:
        deadline_ms  : task's soft deadline in ms from submission
        elapsed_ms   : milliseconds since task was submitted
        tier         : workload tier (1=CRITICAL, 2=URGENT, 3=IMPORTANT, 4=BG)
        priority     : user-specified priority override (1=high, 2=normal, 3=low)

    Returns:
        Urgency score ∈ (0, ∞]
    """
    remaining_ms = deadline_ms - elapsed_ms
    if remaining_ms <= 0:
        return math.inf

    tier_w     = TIER_WEIGHTS.get(tier, 1.0)
    priority_w = {1: 1.5, 2: 1.0, 3: 0.7}.get(priority, 1.0)
    fraction   = remaining_ms / max(1.0, float(deadline_ms))
    return (tier_w * priority_w) / max(1e-4, fraction)


def compute_urgency_vector(tasks: list) -> list[tuple]:
    """
    Compute urgency for a list of tasks and return sorted (urgency, task) pairs.
    Used for queue ordering and preemption selection.
    """
    scored = [(t.compute_urgency(), t) for t in tasks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# -- Interference Admission -----------------------------------------------------

@dataclass
class AdmissionDecision:
    admit:       bool
    reason:      str
    risk_score:  float         # 0.0=safe, 1.0=maximum interference risk
    budget_used: float         # fraction of most-constrained task's deadline budget consumed
    details:     list[str] = field(default_factory=list)


def check_interference_admission(
    candidate_class: str,
    candidate_deadline_ms: int,
    running_tasks: list,         # list of Task objects
    confidence_map: Optional[dict] = None,
) -> AdmissionDecision:
    """
    Core admission check: can we safely admit `candidate_class`
    alongside all currently running tasks?

    Algorithm:
    1. For each running task, look up interference[candidate][running_class].
    2. Apply confidence-based safety margin.
    3. Reject if:
       a. ipc_degradation > INTERFERENCE_IPC_THRESHOLD
       b. lat_overhead > DEADLINE_BUDGET_FRACTION × running_deadline_remaining
       c. candidate's own deadline would be violated by accumulated overhead
    4. Accept otherwise.

    Returns AdmissionDecision with detailed reasoning.
    """
    if len(running_tasks) >= MAX_CONCURRENT_TASKS:
        return AdmissionDecision(
            admit=False, reason="max_concurrency_reached",
            risk_score=1.0, budget_used=1.0,
            details=[f"Running={len(running_tasks)} ≥ MAX={MAX_CONCURRENT_TASKS}"]
        )

    if candidate_class not in INTERFERENCE_MATRIX:
        # Unknown class -- conservative: admit only if few tasks running
        if len(running_tasks) <= 2:
            return AdmissionDecision(admit=True, reason="ok_unknown_class",
                                     risk_score=0.3, budget_used=0.0)
        return AdmissionDecision(admit=False, reason="unknown_class_conservative",
                                 risk_score=0.6, budget_used=0.0)

    max_risk   = 0.0
    max_budget = 0.0
    details    = []
    total_overhead_on_candidate = 0.0

    for rt in running_tasks:
        rc = getattr(rt, 'workload_class', 'unknown')
        if rc not in INTERFERENCE_MATRIX.get(candidate_class, {}):
            continue

        ipc_deg, lat_ms = INTERFERENCE_MATRIX[candidate_class][rc]

        # Confidence-based safety margin
        conf = (confidence_map or {}).get((candidate_class, rc), 0.5)
        safety = 1.0 + (1.0 - conf) * CONFIDENCE_SAFETY_FACTOR
        effective_lat = lat_ms * safety

        # Check: IPC degradation threshold
        if ipc_deg > INTERFERENCE_IPC_THRESHOLD:
            return AdmissionDecision(
                admit=False,
                reason=f"ipc_threshold:{candidate_class}→{rc}:{ipc_deg:.2f}>{INTERFERENCE_IPC_THRESHOLD}",
                risk_score=ipc_deg, budget_used=0.0,
                details=[f"IPC degradation {ipc_deg:.1%} exceeds threshold {INTERFERENCE_IPC_THRESHOLD:.1%}"]
            )

        # Check: deadline budget for running task
        rt_remaining = rt.deadline_remaining_ms()
        budget_fraction = effective_lat / max(1.0, rt_remaining) if rt_remaining > 0 else 1.0

        if budget_fraction > DEADLINE_BUDGET_FRACTION:
            return AdmissionDecision(
                admit=False,
                reason=f"deadline_squeeze:{rt.task_id}:{rt_remaining:.0f}ms_left",
                risk_score=ipc_deg,
                budget_used=budget_fraction,
                details=[
                    f"Adding {effective_lat:.1f}ms to {rt.task_id} which has {rt_remaining:.0f}ms left",
                    f"Budget fraction {budget_fraction:.1%} > limit {DEADLINE_BUDGET_FRACTION:.1%}"
                ]
            )

        # Accumulate overhead on candidate itself (reverse interference)
        if rc in INTERFERENCE_MATRIX and candidate_class in INTERFERENCE_MATRIX[rc]:
            rev_ipc, rev_lat = INTERFERENCE_MATRIX[rc][candidate_class]
            total_overhead_on_candidate += rev_lat * safety

        max_risk   = max(max_risk,   ipc_deg)
        max_budget = max(max_budget, budget_fraction)
        details.append(f"{rc}→{candidate_class}: ipc_deg={ipc_deg:.2f} lat={effective_lat:.1f}ms")

    # Check: candidate's own deadline feasibility under accumulated overhead
    if total_overhead_on_candidate > 0.60 * candidate_deadline_ms:
        return AdmissionDecision(
            admit=False,
            reason=f"candidate_infeasible:overhead={total_overhead_on_candidate:.0f}ms>60%_of_{candidate_deadline_ms}ms",
            risk_score=max_risk, budget_used=max_budget,
            details=[f"Accumulated overhead {total_overhead_on_candidate:.0f}ms exceeds 60% of deadline {candidate_deadline_ms}ms"]
        )

    return AdmissionDecision(
        admit=True, reason="ok",
        risk_score=round(max_risk, 4),
        budget_used=round(max_budget, 4),
        details=details,
    )


# -- Jain's Fairness Index -----------------------------------------------------

def jains_fairness_index(allocations: list[float]) -> float:
    """
    Jain's Fairness Index: measures how equitably resources are distributed.

    JFI = (Σxᵢ)² / (n × Σxᵢ²)

    Range: [1/n, 1.0] where 1.0 = perfect fairness.

    In MOSAIC context, `allocations` = list of deadline-hit-rates per class.
    A JFI close to 1.0 means all workload classes are meeting their deadlines
    at similar rates -- no class is being systematically starved.
    """
    n = len(allocations)
    if n == 0:
        return 1.0
    sum_x  = sum(allocations)
    sum_x2 = sum(x * x for x in allocations)
    if sum_x2 == 0:
        return 1.0
    return (sum_x ** 2) / (n * sum_x2)


# -- Starvation Guard ----------------------------------------------------------

def detect_starvation(queued_tasks: list,
                      starvation_mult: float = STARVATION_TIMEOUT_MULT) -> list:
    """
    Identifies tasks in the queue that have been waiting so long they are
    at risk of starvation regardless of deadline miss.

    A task is "starving" if its queue age exceeds starvation_mult × deadline.
    Starving tasks get an urgency boost: U = ∞ (same as past-deadline).

    Returns list of task IDs that are starving.
    """
    starving = []
    for task in queued_tasks:
        age_ms = task.age_ms()
        if age_ms > starvation_mult * task.deadline_ms:
            starving.append(task.task_id)
    return starving


# -- Energy DVFS Trigger --------------------------------------------------------

def should_throttle(current_watts: float, power_cap: float) -> bool:
    return current_watts > power_cap * DVFS_POWER_THRESHOLD


def select_throttle_target(running_tasks: list) -> Optional[object]:
    """
    Select the lowest-urgency running task for DVFS throttling.
    Ties broken by tier (higher tier = protected).
    """
    if not running_tasks:
        return None
    return min(running_tasks, key=lambda t: (t.compute_urgency(), -t.tier))
