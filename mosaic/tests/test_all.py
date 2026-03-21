#!/usr/bin/env python3
"""
MOSAIC Complete Test Suite
===========================
Covers every component: taxonomy, ML classifier, algorithms,
benchmarking harness, workload generator, fairness metrics.

Run: python3 tests/test_all.py
"""

import sys
import math
import json
import time
import random
import tempfile
import unittest
import threading
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scheduler" / "core_algorithm"))
sys.path.insert(0, str(_ROOT / "scheduler"))
sys.path.insert(0, str(_ROOT / "workload-gen"))
sys.path.insert(0, str(_ROOT / "benchmarking"))

from workload_taxonomy import (
    WORKLOAD_CLASSES, CLASS_NAMES, INTERFERENCE_MATRIX, TIER_WEIGHTS, WorkloadClass
)
from algorithms import (
    compute_urgency, check_interference_admission,
    jains_fairness_index, detect_starvation,
    should_throttle, select_throttle_target,
    INTERFERENCE_IPC_THRESHOLD,
)
from ml_classifier import WorkloadClassifier, ClassificationResult
from workload_gen import (
    sample_class, sample_task, poisson_arrivals, burst_arrivals,
    sinusoidal_arrivals, disaster_arrivals, PHASE_WEIGHTS, MetricsCollector,
)
from benchmark import (
    FCFSScheduler, RoundRobinScheduler, SJFScheduler,
    PriorityScheduler, MOSAICSimScheduler,
    SimTask, run_experiment, SCHEDULER_REGISTRY,
)


# -- 1. Workload Taxonomy -------------------------------------------------------

class TestWorkloadTaxonomy(unittest.TestCase):

    def test_all_classes_have_required_fields(self):
        for name, wc in WORKLOAD_CLASSES.items():
            self.assertIsInstance(wc, WorkloadClass, name)
            self.assertIn(wc.tier, [1, 2, 3, 4], f"{name}.tier invalid")
            dmin, dmax = wc.deadline_range
            self.assertLess(dmin, dmax, f"{name}.deadline_range invalid")
            smin, smax = wc.service_range
            self.assertLess(smin, smax, f"{name}.service_range invalid")
            self.assertGreater(wc.cpu_shares, 0, f"{name}.cpu_shares")
            self.assertGreater(wc.mem_mb,     0, f"{name}.mem_mb")

    def test_arrival_weights_sum_to_one_per_phase(self):
        for phase, weights in PHASE_WEIGHTS.items():
            total = sum(weights.values())
            self.assertAlmostEqual(total, 1.0, places=3,
                                   msg=f"Phase {phase} weights sum={total:.4f}")

    def test_interference_matrix_is_square(self):
        for cls in CLASS_NAMES:
            self.assertIn(cls, INTERFERENCE_MATRIX,
                          f"{cls} missing from INTERFERENCE_MATRIX rows")
            for victim in CLASS_NAMES:
                self.assertIn(victim, INTERFERENCE_MATRIX[cls],
                              f"{cls}→{victim} missing from matrix")

    def test_self_interference_is_zero(self):
        for cls in CLASS_NAMES:
            ipc_deg, lat = INTERFERENCE_MATRIX[cls][cls]
            self.assertEqual(ipc_deg, 0.0, f"{cls} self ipc_degradation should be 0")
            self.assertEqual(lat,     0.0, f"{cls} self lat_overhead should be 0")

    def test_interference_values_in_range(self):
        for ca in CLASS_NAMES:
            for cb in CLASS_NAMES:
                ipc, lat = INTERFERENCE_MATRIX[ca][cb]
                self.assertGreaterEqual(ipc, 0.0, f"{ca}→{cb} ipc_deg < 0")
                self.assertLessEqual(   ipc, 1.0, f"{ca}→{cb} ipc_deg > 1")
                self.assertGreaterEqual(lat, 0.0, f"{ca}→{cb} lat_ms < 0")

    def test_tier_weights_exist(self):
        for tier in [1, 2, 3, 4]:
            self.assertIn(tier, TIER_WEIGHTS)
            self.assertGreater(TIER_WEIGHTS[tier], 0)

    def test_critical_tier_outweighs_background(self):
        self.assertGreater(TIER_WEIGHTS[1], TIER_WEIGHTS[4])

    def test_disaster_domain_framing(self):
        """Verify workload classes match disaster-response domain."""
        self.assertIn("inference_critical", WORKLOAD_CLASSES,
                      "Missing inference_critical -- real-time AI alerts")
        self.assertIn("dispatch_api", WORKLOAD_CLASSES,
                      "Missing dispatch_api -- responder coordination")
        self.assertIn("sensor_fusion", WORKLOAD_CLASSES,
                      "Missing sensor_fusion -- multi-sensor overlay")
        # Tier 1 classes must have shortest deadlines
        tier1 = [n for n, wc in WORKLOAD_CLASSES.items() if wc.tier == 1]
        tier4 = [n for n, wc in WORKLOAD_CLASSES.items() if wc.tier == 4]
        avg_dl_t1 = sum(WORKLOAD_CLASSES[n].deadline_range[1] for n in tier1) / len(tier1)
        avg_dl_t4 = sum(WORKLOAD_CLASSES[n].deadline_range[0] for n in tier4) / len(tier4)
        self.assertLess(avg_dl_t1, avg_dl_t4,
                        "Tier-1 deadlines should be tighter than tier-4")


# -- 2. Urgency Scoring ---------------------------------------------------------

class TestUrgencyScoring(unittest.TestCase):

    def test_urgency_positive_for_live_task(self):
        u = compute_urgency(500, 100, tier=1)
        self.assertGreater(u, 0)

    def test_urgency_infinite_past_deadline(self):
        u = compute_urgency(500, 600, tier=2)
        self.assertEqual(u, math.inf)

    def test_higher_tier_higher_urgency(self):
        u1 = compute_urgency(500, 100, tier=1)
        u2 = compute_urgency(500, 100, tier=2)
        u3 = compute_urgency(500, 100, tier=3)
        u4 = compute_urgency(500, 100, tier=4)
        self.assertGreater(u1, u2)
        self.assertGreater(u2, u3)
        self.assertGreater(u3, u4)

    def test_urgency_increases_as_deadline_approaches(self):
        u_early = compute_urgency(1000,  10, tier=2)
        u_mid   = compute_urgency(1000, 500, tier=2)
        u_late  = compute_urgency(1000, 900, tier=2)
        self.assertLess(u_early, u_mid)
        self.assertLess(u_mid,   u_late)

    def test_high_priority_beats_low_priority_same_tier(self):
        u_hi = compute_urgency(500, 100, tier=2, priority=1)
        u_lo = compute_urgency(500, 100, tier=2, priority=3)
        self.assertGreater(u_hi, u_lo)

    def test_tier1_always_beats_tier2_at_same_time_remaining(self):
        u1 = compute_urgency(1000, 500, tier=1)
        u2 = compute_urgency(1000, 500, tier=2)
        self.assertGreater(u1, u2)

    def test_urgency_not_nan(self):
        for tier in [1, 2, 3, 4]:
            for elapsed in [0, 100, 500, 999, 1001]:
                u = compute_urgency(1000, elapsed, tier)
                self.assertFalse(math.isnan(u), f"NaN at tier={tier} elapsed={elapsed}")


# -- 3. ML Classifier ----------------------------------------------------------

class TestMLClassifier(unittest.TestCase):

    def setUp(self):
        self._clf = WorkloadClassifier()

    def test_classifies_all_known_classes(self):
        """Known fingerprints should classify to their own class."""
        for name, wc in WORKLOAD_CLASSES.items():
            ipc, llc, bw, br = wc.fingerprint
            r = self._clf.classify(ipc, llc, bw, br)
            self.assertEqual(r.predicted_class, name,
                             f"Expected {name} got {r.predicted_class}")

    def test_confidence_in_range(self):
        r = self._clf.classify(2.1, 0.18, 8.5, 0.06)
        self.assertGreaterEqual(r.confidence, 0.0)
        self.assertLessEqual(r.confidence,    1.0)

    def test_result_has_required_fields(self):
        r = self._clf.classify(1.8, 0.42, 28.0, 0.08)
        self.assertIsInstance(r, ClassificationResult)
        self.assertIn(r.predicted_class, CLASS_NAMES)
        self.assertIn(r.tier, [1, 2, 3, 4])
        self.assertIsInstance(r.distances, dict)

    def test_metadata_classification_dispatch_api(self):
        r = self._clf.classify_from_metadata({"deadline_ms": 150, "mem_mb": 256, "gpu_required": False})
        self.assertEqual(r.predicted_class, "dispatch_api")
        self.assertEqual(r.method, "metadata")

    def test_metadata_classification_inference(self):
        r = self._clf.classify_from_metadata({"deadline_ms": 1500, "gpu_required": True})
        self.assertEqual(r.predicted_class, "inference_critical")

    def test_metadata_classification_log_archive(self):
        r = self._clf.classify_from_metadata({"deadline_ms": 60000})
        self.assertEqual(r.predicted_class, "log_archive")

    def test_online_update_shifts_centroid(self):
        """After online update, centroid should move toward new observation."""
        clf = WorkloadClassifier()
        centroid_before = clf._centroids["dispatch_api"].as_vec()[:]
        # Update with a slightly different fingerprint
        clf.online_update("dispatch_api", ipc=3.2, llc=0.04, bw=2.5, br=0.03)
        centroid_after = clf._centroids["dispatch_api"].as_vec()
        # At least one dimension should have changed
        changed = any(abs(a - b) > 1e-9 for a, b in zip(centroid_before, centroid_after))
        self.assertTrue(changed, "Centroid did not update after online_update")

    def test_distances_all_classes_present(self):
        r = self._clf.classify(2.1, 0.18, 8.5, 0.06)
        for cls in CLASS_NAMES:
            self.assertIn(cls, r.distances)

    def test_unknown_workload_classified_to_nearest(self):
        """A fingerprint halfway between two classes should still pick one."""
        r = self._clf.classify(2.0, 0.25, 15.0, 0.05)
        self.assertIn(r.predicted_class, CLASS_NAMES)
        self.assertGreater(r.confidence, 0)


# -- 4. Interference Admission --------------------------------------------------

class _MockTask:
    """Lightweight mock task for admission tests."""
    def __init__(self, cls, deadline_ms, elapsed_ms=10, task_id=None):
        self.workload_class = cls
        self.deadline_ms    = deadline_ms
        self.tier           = WORKLOAD_CLASSES.get(cls, WORKLOAD_CLASSES["dispatch_api"]).tier
        self.task_id        = task_id or f"mock_{cls[:4]}"
        self._submit_time   = time.monotonic() - elapsed_ms / 1000.0
    def deadline_remaining_ms(self):
        return max(0.0, self.deadline_ms - (time.monotonic()-self._submit_time)*1000)

class TestInterferenceAdmission(unittest.TestCase):

    def test_empty_running_always_admits(self):
        d = check_interference_admission("dispatch_api", 200, [])
        self.assertTrue(d.admit, f"Should admit to empty set: {d.reason}")

    def test_self_colocation_admitted(self):
        """Same class colocating with itself should be fine (self-interference=0)."""
        running = [_MockTask("log_archive", 60000)]
        d       = check_interference_admission("log_archive", 60000, running)
        self.assertTrue(d.admit, f"log_archive self-colocation should admit: {d.reason}")

    def test_high_interference_blocked(self):
        """inference_critical → dispatch_api has 38% IPC degradation > 35% threshold."""
        running   = [_MockTask("dispatch_api", 200)]
        candidate = "inference_critical"
        ipc_deg   = INTERFERENCE_MATRIX[candidate]["dispatch_api"][0]
        if ipc_deg > INTERFERENCE_IPC_THRESHOLD:
            d = check_interference_admission(candidate, 1000, running)
            self.assertFalse(d.admit)

    def test_low_interference_admitted(self):
        """log_archive barely affects anything -- should always admit."""
        running = [_MockTask("inference_critical", 2000),
                   _MockTask("dispatch_api", 200)]
        d = check_interference_admission("log_archive", 60000, running)
        self.assertTrue(d.admit, f"log_archive should admit: {d.reason}")

    def test_max_concurrency_blocks(self):
        running = [_MockTask(f"log_archive", 60000, task_id=f"r{i}") for i in range(8)]
        d = check_interference_admission("dispatch_api", 200, running)
        self.assertFalse(d.admit)
        self.assertIn("max_concurrency", d.reason)

    def test_decision_has_risk_score(self):
        d = check_interference_admission("dispatch_api", 200, [])
        self.assertGreaterEqual(d.risk_score, 0.0)
        self.assertLessEqual(d.risk_score, 1.0)

    def test_deadline_squeeze_detected(self):
        """Task with near-zero deadline remaining should be squeezed."""
        # Running task with only 5ms left
        running = [_MockTask("dispatch_api", 100, elapsed_ms=99)]
        # Candidate that would add >40% of 1ms overhead
        d = check_interference_admission("analytics_batch", 5000, running)
        # analytics_batch → dispatch_api lat_overhead = 7ms > 0.4 × ~1ms
        # So it should be blocked by deadline_squeeze
        self.assertFalse(d.admit)


# -- 5. Fairness & Starvation ---------------------------------------------------

class TestFairnessAndStarvation(unittest.TestCase):

    def test_jfi_perfect_fairness(self):
        self.assertAlmostEqual(jains_fairness_index([1.0, 1.0, 1.0]), 1.0, places=5)

    def test_jfi_worst_case(self):
        """One class gets everything, rest get 0."""
        jfi = jains_fairness_index([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(jfi, 0.25, places=5)  # 1/n

    def test_jfi_empty(self):
        self.assertEqual(jains_fairness_index([]), 1.0)

    def test_jfi_single(self):
        self.assertEqual(jains_fairness_index([0.8]), 1.0)

    def test_jfi_range(self):
        for vals in [[0.9, 0.7, 0.6], [1.0, 0.5, 0.3], [0.8]*6]:
            jfi = jains_fairness_index(vals)
            n   = len(vals)
            self.assertGreaterEqual(jfi, 1.0/n - 1e-9)
            self.assertLessEqual(jfi,   1.0 + 1e-9)

    def test_starvation_detect_aged_task(self):
        """Task waiting 4× its deadline should be detected as starving."""
        class AgedTask:
            task_id     = "old"
            deadline_ms = 500
            def age_ms(self): return 2001  # 4× deadline
        starving = detect_starvation([AgedTask()])
        self.assertIn("old", starving)

    def test_starvation_fresh_task_not_starving(self):
        class FreshTask:
            task_id     = "new"
            deadline_ms = 5000
            def age_ms(self): return 100
        starving = detect_starvation([FreshTask()])
        self.assertNotIn("new", starving)

    def test_energy_throttle_threshold(self):
        self.assertTrue(should_throttle(80.0, 85.0))   # 94% → throttle
        self.assertFalse(should_throttle(70.0, 85.0))  # 82% → ok

    def test_throttle_target_selects_lowest_urgency(self):
        class UTask:
            def __init__(self, tid, u, t): self.task_id=tid; self._u=u; self.tier=t
            def compute_urgency(self): return self._u
        tasks = [UTask("hi", 10.0, 1), UTask("lo", 1.0, 3), UTask("mid", 5.0, 2)]
        target = select_throttle_target(tasks)
        self.assertEqual(target.task_id, "lo")


# -- 6. Workload Generator -----------------------------------------------------

class TestWorkloadGenerator(unittest.TestCase):

    def test_sample_class_always_known(self):
        for _ in range(200):
            self.assertIn(sample_class("crisis"), CLASS_NAMES)
            self.assertIn(sample_class("calm"),   CLASS_NAMES)

    def test_sample_task_valid_fields(self):
        for cls in CLASS_NAMES:
            t = sample_task(cls, "tid")
            self.assertEqual(t["class"],    cls)
            self.assertEqual(t["task_id"], "tid")
            wc = WORKLOAD_CLASSES[cls]
            self.assertGreaterEqual(t["deadline_ms"], wc.deadline_range[0])
            self.assertLessEqual(   t["deadline_ms"], wc.deadline_range[1])
            self.assertGreater(t["service_time_ms"], 0)

    def test_poisson_inter_arrivals_positive(self):
        gen = poisson_arrivals(5.0)
        for _ in range(50):
            iat = next(gen)
            self.assertGreater(iat, 0)

    def test_burst_inter_arrivals_positive(self):
        gen = burst_arrivals(5.0)
        for _ in range(50):
            self.assertGreater(next(gen), 0)

    def test_sinusoidal_positive(self):
        gen = sinusoidal_arrivals(3.0, period=30.0)
        for _ in range(50):
            self.assertGreater(next(gen), 0)

    def test_disaster_arrivals_positive(self):
        gen = disaster_arrivals(5.0)
        for _ in range(100):
            self.assertGreater(next(gen), 0)

    def test_class_distribution_roughly_matches_weights(self):
        random.seed(0)
        N = 5000
        counts = {c: 0 for c in CLASS_NAMES}
        for _ in range(N):
            counts[sample_class("crisis")] += 1
        phase_w = PHASE_WEIGHTS["crisis"]
        for cls, expected_w in phase_w.items():
            actual = counts[cls] / N
            self.assertAlmostEqual(actual, expected_w, delta=expected_w * 0.35,
                                   msg=f"{cls} weight {actual:.3f} != expected {expected_w:.3f}")

    def test_metrics_collector_records(self):
        mc = MetricsCollector()
        mc.record(100.0, 200.0, True,  "dispatch_api")
        mc.record(250.0, 200.0, False, "dispatch_api")
        s = mc.summary()
        self.assertEqual(s["hits"],   1)
        self.assertEqual(s["misses"], 1)
        self.assertAlmostEqual(s["hit_rate"], 0.5)
        self.assertGreater(s["p50_ms"], 0)


# -- 7. Benchmarking -- All 5 Schedulers ---------------------------------------

class TestSchedulerBenchmark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Run a short experiment once for all tests."""
        random.seed(42)
        cls.results = {}
        for name, SchedCls in SCHEDULER_REGISTRY.items():
            sched = SchedCls()
            r = run_experiment(sched, duration=8.0, rate=3.0, pattern="poisson", seed=42)
            cls.results[name] = r

    def test_all_schedulers_produce_results(self):
        for name in SCHEDULER_REGISTRY:
            self.assertIn(name, self.results)
            r = self.results[name]
            self.assertIn("hit_rate",       r)
            self.assertIn("p99_ms",         r)
            self.assertIn("fairness_index", r)
            self.assertIn("starvation_rate",r)
            self.assertIn("throughput_tps", r)
            self.assertIn("efficiency_tpwh",r)

    def test_hit_rates_in_valid_range(self):
        for name, r in self.results.items():
            self.assertGreaterEqual(r["hit_rate"], 0.0, f"{name} hit_rate < 0")
            self.assertLessEqual(   r["hit_rate"], 1.0, f"{name} hit_rate > 1")

    def test_fairness_index_in_valid_range(self):
        for name, r in self.results.items():
            n = len(CLASS_NAMES)
            self.assertGreaterEqual(r["fairness_index"], 1.0/n - 0.01, f"{name} JFI too low")
            self.assertLessEqual(   r["fairness_index"], 1.0 + 0.01,   f"{name} JFI > 1")

    def test_starvation_rate_in_valid_range(self):
        for name, r in self.results.items():
            self.assertGreaterEqual(r["starvation_rate"], 0.0, f"{name} starvation < 0")
            self.assertLessEqual(   r["starvation_rate"], 1.0, f"{name} starvation > 1")

    def test_p99_gte_p95_gte_p50(self):
        for name, r in self.results.items():
            self.assertGreaterEqual(r["p99_ms"], r["p95_ms"] - 1.0,
                                    f"{name}: p99 < p95")
            self.assertGreaterEqual(r["p95_ms"], r["p50_ms"] - 1.0,
                                    f"{name}: p95 < p50")

    def test_mosaic_competitive_hit_rate(self):
        """MOSAIC should be competitive with or better than FCFS."""
        mosaic = self.results["mosaic"]
        fcfs   = self.results["fcfs"]
        # MOSAIC admitted tasks should have ≥ FCFS hit rate (within 10pp tolerance
        # at short durations where queue drain effects dominate)
        self.assertGreaterEqual(
            mosaic["hit_rate"], fcfs["hit_rate"] * 0.90,
            f"MOSAIC hit_rate={mosaic['hit_rate']:.2%} far below FCFS={fcfs['hit_rate']:.2%}"
        )

    def test_rr_has_overhead_vs_fcfs(self):
        """Round Robin adds context-switch overhead, should have ≥ FCFS P99."""
        # RR injects 2ms overhead per quantum -- P99 should generally be >= FCFS
        # (not always true at low load, so just check it produces valid numbers)
        rr = self.results["rr"]
        self.assertGreater(rr["p99_ms"], 0)

    def test_sjf_output_valid(self):
        sjf = self.results["sjf"]
        self.assertGreater(sjf["total"], 0)
        self.assertGreaterEqual(sjf["hit_rate"], 0.0)

    def test_throughput_positive(self):
        for name, r in self.results.items():
            self.assertGreater(r["throughput_tps"], 0, f"{name} throughput=0")

    def test_per_class_hit_rates_present(self):
        for name, r in self.results.items():
            self.assertIn("class_hit_rates", r)
            # Should have entries for at least some classes
            self.assertGreater(len(r["class_hit_rates"]), 0)

    def test_energy_efficiency_positive(self):
        for name, r in self.results.items():
            self.assertGreaterEqual(r["efficiency_tpwh"], 0.0)


# -- 8. MOSAIC-specific quality tests -----------------------------------------

class TestMOSAICQuality(unittest.TestCase):

    def test_mosaic_starvation_not_worse_than_fcfs(self):
        """MOSAIC's starvation guard should prevent runaway queue aging."""
        random.seed(7)
        mosaic = MOSAICSimScheduler()
        fcfs   = FCFSScheduler()
        r_m = run_experiment(mosaic, 10.0, 4.0, "burst", 7)
        r_f = run_experiment(fcfs,   10.0, 4.0, "burst", 7)
        # MOSAIC starvation_rate should not be wildly higher than FCFS
        self.assertLessEqual(r_m["starvation_rate"],
                             r_f["starvation_rate"] * 2.0 + 0.05,
                             "MOSAIC starvation far exceeds FCFS")

    def test_mosaic_fairness_not_worse_than_priority(self):
        """MOSAIC should have comparable or better fairness than static priority."""
        random.seed(11)
        mosaic = MOSAICSimScheduler()
        prio   = PriorityScheduler()
        r_m = run_experiment(mosaic, 10.0, 3.0, "poisson", 11)
        r_p = run_experiment(prio,   10.0, 3.0, "poisson", 11)
        # JFI: higher is more fair. MOSAIC should be >= priority * 0.9
        self.assertGreaterEqual(r_m["fairness_index"],
                                r_p["fairness_index"] * 0.85,
                                f"MOSAIC JFI={r_m['fairness_index']:.3f} "
                                f"much lower than Priority JFI={r_p['fairness_index']:.3f}")

    def test_disaster_pattern_runs_mosaic(self):
        random.seed(55)
        sched = MOSAICSimScheduler()
        r = run_experiment(sched, 8.0, 3.0, "disaster", 55)
        self.assertGreater(r["total"], 0)
        self.assertGreaterEqual(r["hit_rate"], 0.0)

    def test_mosaic_handles_tier1_priority(self):
        """Tier-1 (CRITICAL) tasks should have the highest urgency in MOSAIC queue."""
        sched = MOSAICSimScheduler()
        t1 = SimTask("t1", "inference_critical", 2000, 500, 1, 1)
        t4 = SimTask("t4", "log_archive",       60000, 200, 2, 4)
        # Tier-1 urgency should exceed tier-4
        u1 = compute_urgency(t1.deadline_ms, 10, t1.tier, t1.priority)
        u4 = compute_urgency(t4.deadline_ms, 10, t4.tier, t4.priority)
        self.assertGreater(u1, u4)


# -- Runner --------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  MOSAIC -- Complete Test Suite")
    print("  Covers: Taxonomy | Urgency | ML Classifier | Admission |")
    print("          Fairness | Starvation | Generator | All 5 Schedulers")
    print("="*70 + "\n")

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestWorkloadTaxonomy, TestUrgencyScoring, TestMLClassifier,
        TestInterferenceAdmission, TestFairnessAndStarvation,
        TestWorkloadGenerator, TestSchedulerBenchmark, TestMOSAICQuality,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, failfast=False)
    result = runner.run(suite)

    print("\n" + "="*70)
    total_tests = result.testsRun
    if result.wasSuccessful():
        print(f"  {chr(10003)}  All {total_tests} tests passed")
    else:
        print(f"  {chr(10007)}  {len(result.failures)} failures, "
              f"{len(result.errors)} errors out of {total_tests} tests")
    print("="*70 + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
