"""
MOSAIC ML Workload Classifier
==============================
Classifies incoming tasks into workload classes using a two-stage approach:

Stage 1 -- Nearest-centroid classification on hardware counter fingerprint vector.
           Fast (O(k) where k=num_classes), no training data needed initially.

Stage 2 -- Online EWC (Exponentially Weighted Centroid) update.
           Each observed task refines the class centroid, improving accuracy
           over time without storing the full dataset.

WOW factor: the classifier is entirely unsupervised at runtime.
            Operators profile each class ONCE offline. The scheduler then
            classifies all future arrivals automatically, even new workload
            variants that drift from the original fingerprint.

Inputs (from perf_event profiler OR estimated from task metadata):
    ipc             : instructions per cycle
    llc_miss_rate   : LLC misses / LLC accesses ∈ [0,1]
    mem_bw_gbs      : memory bandwidth GB/s
    branch_miss_rate: branch misses / total branches ∈ [0,1]

Output: workload_class name + confidence score ∈ [0,1]
"""

from __future__ import annotations

import math
import json
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

from workload_taxonomy import (
    WORKLOAD_CLASSES, CLASS_NAMES, WorkloadClass
)

# Normalisation scales -- approximate domain max per dimension
SCALE = {
    "ipc":              4.0,
    "llc_miss_rate":    1.0,
    "mem_bw_gbs":      50.0,
    "branch_miss_rate": 1.0,
}

# Dimension weights for classification (LLC miss + MemBW dominate interference)
DIM_WEIGHTS = {
    "ipc":              1.0,
    "llc_miss_rate":    2.5,
    "mem_bw_gbs":       2.0,
    "branch_miss_rate": 0.8,
}


@dataclass
class ClassCentroid:
    name:             str
    ipc:              float
    llc_miss_rate:    float
    mem_bw_gbs:       float
    branch_miss_rate: float
    n_samples:        int   = 1
    last_updated:     float = field(default_factory=time.time)

    def as_vec(self) -> list[float]:
        return [self.ipc, self.llc_miss_rate, self.mem_bw_gbs, self.branch_miss_rate]

    def update(self, ipc: float, llc: float, bw: float, br: float,
               alpha: float = 0.15) -> None:
        """Exponentially weighted centroid update -- online learning, O(1)."""
        self.ipc             = (1 - alpha) * self.ipc             + alpha * ipc
        self.llc_miss_rate   = (1 - alpha) * self.llc_miss_rate   + alpha * llc
        self.mem_bw_gbs      = (1 - alpha) * self.mem_bw_gbs      + alpha * bw
        self.branch_miss_rate= (1 - alpha) * self.branch_miss_rate + alpha * br
        self.n_samples      += 1
        self.last_updated    = time.time()


@dataclass
class ClassificationResult:
    predicted_class: str
    confidence:      float   # 0.0–1.0
    tier:            int
    distances:       dict[str, float]  # all class distances for inspection
    method:          str     # "fingerprint" | "metadata" | "default"


class WorkloadClassifier:
    """
    Two-stage ML workload classifier for MOSAIC.

    Usage:
        clf = WorkloadClassifier()
        result = clf.classify(ipc=2.1, llc_miss_rate=0.08, mem_bw_gbs=3.5, branch_miss_rate=0.04)
        print(result.predicted_class, result.confidence)

        # Or classify from task metadata alone (no perf counters)
        result = clf.classify_from_metadata({"deadline_ms": 150, "mem_mb": 256, "gpu": False})
    """

    CENTROID_PATH = Path(__file__).parent.parent.parent / "data" / "centroids.json"

    def __init__(self):
        self._centroids: dict[str, ClassCentroid] = {}
        self._load_centroids()

    # -- Centroid Management ------------------------------------------------

    def _load_centroids(self) -> None:
        """Load centroids from disk if available, otherwise seed from taxonomy."""
        if self.CENTROID_PATH.exists():
            try:
                data = json.loads(self.CENTROID_PATH.read_text(encoding="utf-8"))
                for name, vals in data.items():
                    self._centroids[name] = ClassCentroid(**vals)
                return
            except (json.JSONDecodeError, TypeError):
                pass
        self._seed_from_taxonomy()

    def _seed_from_taxonomy(self) -> None:
        """Seed centroids from the hardcoded fingerprint vectors in workload_taxonomy."""
        for name, wc in WORKLOAD_CLASSES.items():
            ipc, llc, bw, br = wc.fingerprint
            self._centroids[name] = ClassCentroid(
                name=name, ipc=ipc, llc_miss_rate=llc,
                mem_bw_gbs=bw, branch_miss_rate=br,
            )

    def save_centroids(self) -> None:
        self.CENTROID_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {name: asdict(c) for name, c in self._centroids.items()}
        self.CENTROID_PATH.write_text(json.dumps(data, indent=2))

    # -- Core Classification ------------------------------------------------

    def _weighted_distance(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Weighted normalised Euclidean distance."""
        dims  = ["ipc", "llc_miss_rate", "mem_bw_gbs", "branch_miss_rate"]
        total = 0.0
        for i, dim in enumerate(dims):
            diff   = (vec_a[i] - vec_b[i]) / SCALE[dim]
            total += DIM_WEIGHTS[dim] * diff * diff
        return math.sqrt(total)

    def classify(self,
                 ipc:              float,
                 llc_miss_rate:    float,
                 mem_bw_gbs:       float,
                 branch_miss_rate: float) -> ClassificationResult:
        """
        Classify a task from its hardware counter fingerprint.
        Returns predicted class + confidence score.
        """
        query = [ipc, llc_miss_rate, mem_bw_gbs, branch_miss_rate]
        distances: dict[str, float] = {}

        for name, centroid in self._centroids.items():
            distances[name] = self._weighted_distance(query, centroid.as_vec())

        # Nearest centroid
        best_class = min(distances, key=distances.__getitem__)
        best_dist  = distances[best_class]

        # Confidence: 1 - (best_dist / mean_dist), softmax-normalised
        mean_dist   = sum(distances.values()) / len(distances)
        raw_conf    = 1.0 - (best_dist / (mean_dist + 1e-9))
        confidence  = max(0.0, min(1.0, raw_conf))

        return ClassificationResult(
            predicted_class = best_class,
            confidence      = round(confidence, 4),
            tier            = WORKLOAD_CLASSES[best_class].tier,
            distances       = {k: round(v, 4) for k, v in distances.items()},
            method          = "fingerprint",
        )

    def classify_from_metadata(self, metadata: dict) -> ClassificationResult:
        """
        Classify from task submission metadata when perf counters are unavailable.
        Uses a rule-based decision tree that matches real-world intuition.

        Rules are ordered by specificity (most specific first).
        """
        deadline   = metadata.get("deadline_ms", 5000)
        mem_mb     = metadata.get("mem_mb", 512)
        gpu        = metadata.get("gpu_required", False)
        tier_hint  = metadata.get("tier", 0)  # optional explicit hint

        # Tier hint overrides (when operator explicitly labels)
        if tier_hint == 1:
            best = "inference_critical"
        elif tier_hint == 4:
            best = "log_archive"
        # Hard deadline + GPU → inference
        elif deadline <= 3000 and gpu:
            best = "inference_critical"
        # Very short deadline → dispatch API
        elif deadline <= 200:
            best = "dispatch_api"
        # Short deadline + moderate memory → sensor fusion
        elif deadline <= 500 and mem_mb <= 1024:
            best = "sensor_fusion"
        # Long deadline + GPU → model update
        elif deadline > 5000 and gpu:
            best = "model_update"
        # Long deadline + high memory → analytics
        elif deadline > 2000 and mem_mb >= 1024:
            best = "analytics_batch"
        # Very long deadline → archival
        elif deadline > 30000:
            best = "log_archive"
        else:
            best = "analytics_batch"

        # Confidence is lower for metadata-only classification
        return ClassificationResult(
            predicted_class = best,
            confidence      = 0.65,
            tier            = WORKLOAD_CLASSES[best].tier,
            distances       = {},
            method          = "metadata",
        )

    def online_update(self, confirmed_class: str,
                      ipc: float, llc: float, bw: float, br: float) -> None:
        """
        Called after a task completes. Updates the centroid for its confirmed class.
        This is the online learning component -- accuracy improves over time.
        """
        if confirmed_class in self._centroids:
            self._centroids[confirmed_class].update(ipc, llc, bw, br)

    def accuracy_report(self) -> dict:
        return {
            name: {
                "n_samples":    c.n_samples,
                "centroid":     c.as_vec(),
                "last_updated": c.last_updated,
            }
            for name, c in self._centroids.items()
        }


# -- Standalone test ------------------------------------------------------------
if __name__ == "__main__":
    clf = WorkloadClassifier()

    test_cases = [
        # (ipc, llc_miss_rate, mem_bw_gbs, branch_miss_rate, expected)
        (1.8, 0.42, 28.0, 0.08, "inference_critical"),
        (2.9, 0.06,  3.2, 0.04, "dispatch_api"),
        (2.1, 0.18,  8.5, 0.06, "sensor_fusion"),
        (1.1, 0.08,  4.0, 0.03, "log_archive"),
    ]
    correct = 0
    for ipc, llc, bw, br, expected in test_cases:
        r = clf.classify(ipc, llc, bw, br)
        ok = "[OK]" if r.predicted_class == expected else "[FAIL]"
        print(f"  {ok} {expected:<22} → {r.predicted_class:<22} conf={r.confidence:.2f}")
        if r.predicted_class == expected:
            correct += 1
    print(f"\nAccuracy: {correct}/{len(test_cases)} = {correct/len(test_cases):.0%}")
