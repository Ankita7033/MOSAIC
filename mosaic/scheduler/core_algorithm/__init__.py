"""MOSAIC core scheduling algorithm package."""
from .workload_taxonomy import WORKLOAD_CLASSES, CLASS_NAMES, INTERFERENCE_MATRIX, TIER_WEIGHTS
from .algorithms import (
    compute_urgency, compute_urgency_vector,
    check_interference_admission, AdmissionDecision,
    jains_fairness_index, detect_starvation,
    should_throttle, select_throttle_target,
)
from .ml_classifier import WorkloadClassifier, ClassificationResult

__all__ = [
    "WORKLOAD_CLASSES", "CLASS_NAMES", "INTERFERENCE_MATRIX", "TIER_WEIGHTS",
    "compute_urgency", "compute_urgency_vector",
    "check_interference_admission", "AdmissionDecision",
    "jains_fairness_index", "detect_starvation",
    "should_throttle", "select_throttle_target",
    "WorkloadClassifier", "ClassificationResult",
]
