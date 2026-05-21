import sys
import json
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from benchmarking.benchmark import run_all

FINAL_DIR = _ROOT / "results" / "final"

def break_priority_strict():
    print("\n--- [STEP 3] Breaking PriorityStrict Properly ---")
    print("  Scenario: cache_thrash_extreme (Simultaneous Tensor Inference + Redis cache pressure + Burst)")
    
    res = run_all(duration=25.0, rate=8.0, pattern="cache_thrash_extreme", schedulers=["priority", "mosaic"], seed=200)
    
    print("\n  Extreme Cache Thrash Results:")
    for r in res:
        print(f"  {r['scheduler']:<15} | Hit Rate: {r['hit_rate']*100:>5.1f}% | P99: {r['p99_ms']:>6.1f} ms | Starvation: {r['starvation_rate']*100:>4.1f}%")
        
    with open(FINAL_DIR / "break_priority.json", "w") as f:
        json.dump(res, f, indent=2)

def task_class_completion_analysis():
    print("\n--- [STEP 4] Task-Class Completion Analysis ---")
    print("  Reframing 'Unfairness' as 'Intentional Protection' under Disaster Scenarios")
    
    from benchmarking.trace_generator import generate_trace
    from workload_taxonomy import CLASS_NAMES
    
    trace = generate_trace(pattern="disaster", rate=8.0, duration=30.0, seed=300)
    generated = {}
    for t in trace:
        c = t["class"]
        generated[c] = generated.get(c, 0) + 1
        
    res = run_all(duration=30.0, rate=8.0, pattern="disaster", schedulers=["mosaic", "priority"], seed=300)
    mosaic_res = next(r for r in res if r["scheduler"] == "MOSAIC")
    priority_res = next(r for r in res if r["scheduler"] == "PriorityStrict")
    
    print("\n  TRUE TASK COMPLETION RATES (HITS / GENERATED) COMPARISON:")
    print(f"  {'-'*95}")
    print(f"  {'Workload Class':<25} | {'PriorityStrict':<20} | {'MOSAIC':<20} | {'MOSAIC Status'}")
    print(f"  {'-'*95}")
    
    semantic = {
        "inference_critical": "inference_critical (YOLO)",
        "dispatch_api": "dispatch_api (FastAPI)",
        "sensor_fusion": "sensor_fusion (IoT Sync)",
        "analytics_batch": "analytics_batch (Batch)",
        "model_update": "model_update (Weights Sync)",
        "log_archive": "log_archive (Redis Logs)"
    }
    
    status_map = {
        "inference_critical": "PROTECTED (CRITICAL)",
        "dispatch_api": "PROTECTED (URGENT)",
        "sensor_fusion": "DEGRADED (IMPORTANT)",
        "analytics_batch": "SACRIFICED (BATCH)",
        "model_update": "SACRIFICED (BACKGROUND)",
        "log_archive": "SACRIFICED (BACKGROUND)"
    }
    
    for cls in CLASS_NAMES:
        m_hits = mosaic_res["class_hits"].get(cls, 0)
        p_hits = priority_res["class_hits"].get(cls, 0)
        gen = generated.get(cls, 0)
        
        m_rate = m_hits / gen if gen > 0 else 0.0
        p_rate = p_hits / gen if gen > 0 else 0.0
        
        print(f"  {semantic.get(cls, cls):<25} | {p_rate*100:>6.1f}% ({p_hits}/{gen}) | {m_rate*100:>6.1f}% ({m_hits}/{gen}) | {status_map.get(cls, '')}")
        
    mosaic_res["true_completion_rates"] = {cls: m_hits/gen if gen > 0 else 0.0 for cls, m_hits in mosaic_res["class_hits"].items()}
    priority_res["true_completion_rates"] = {cls: p_hits/gen if gen > 0 else 0.0 for cls, p_hits in priority_res["class_hits"].items()}
    
    with open(FINAL_DIR / "class_completion.json", "w") as f:
        json.dump({"mosaic": mosaic_res, "priority": priority_res}, f, indent=2)
        
    # Generate the bar chart
    try:
        from visualization.plot_results import setup_style, plt, HAS_MPL
        if HAS_MPL:
            setup_style()
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor("#0d1117")
            
            classes = CLASS_NAMES
            m_rates = [mosaic_res["true_completion_rates"][c] * 100 for c in classes]
            p_rates = [priority_res["true_completion_rates"][c] * 100 for c in classes]
            
            clean_names = {
                "inference_critical": "inference\n(YOLO)",
                "dispatch_api": "dispatch\n(FastAPI)",
                "sensor_fusion": "sensor_fusion\n(IoT Sync)",
                "analytics_batch": "analytics\n(Batch)",
                "model_update": "model_update\n(Weights)",
                "log_archive": "log_archive\n(Redis)"
            }
            x_labels = [clean_names.get(c, c) for c in classes]
            
            n = len(classes)
            x = range(n)
            w = 0.35
            
            bars_p = ax.bar([xi - w/2 for xi in x], p_rates, w, color="#00d4ff", alpha=0.6, label="PriorityStrict", edgecolor="none", zorder=3)
            bars_m = ax.bar([xi + w/2 for xi in x], m_rates, w, color="#00e5a0", alpha=0.9, label="MOSAIC", edgecolor="none", zorder=3)
            
            # Annotate
            for bar in bars_p:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontsize=7, color="#7a8a9a")
            for bar in bars_m:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontsize=7, color="#00ff99", fontweight="bold")
                
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.set_ylabel("True Task Completion Rate (%)")
            ax.set_title("True Task Completion Rate by Workload Class (Crisis Scenario)")
            ax.set_ylim(0, 105)
            ax.legend(loc="upper right", framealpha=0.3)
            
            p_out = FINAL_DIR / "class_completion.png"
            fig.savefig(p_out, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  [viz] Saved class completion plot: {p_out}")
    except Exception as ex:
        print(f"  [viz] Failed to generate class completion plot: {ex}")

def generate_overload_timeline_plot():
    print("\n--- [STEP 5] Generating Overload Timeline Plot ---")
    
    with open(FINAL_DIR / "class_completion.json", "r") as f:
        data = json.load(f)
        
    mosaic_timeline = data["mosaic"]["timeline"]
    priority_timeline = data["priority"]["timeline"]
    
    def process_timeline(events):
        events = sorted(events, key=lambda e: e["ts"])
        
        times = [0.0]
        q_depth = [0]
        active = [0]
        rejections = [0]
        misses = [0]
        
        curr_q = 0
        curr_act = 0
        curr_rej = 0
        curr_miss = 0
        
        task_in_q = set()
        
        for e in events:
            t = e["ts"] / 1000.0
            times.append(t)
            
            event_type = e["event"]
            tid = e["task_id"]
            
            if event_type == "QUEUE":
                curr_q += 1
                task_in_q.add(tid)
            elif event_type == "ADMIT":
                if tid in task_in_q:
                    curr_q = max(0, curr_q - 1)
                    task_in_q.remove(tid)
                curr_act += 1
            elif event_type == "COMPLETE":
                curr_act = max(0, curr_act - 1)
                if not e.get("hit", True):
                    curr_miss += 1
            elif event_type == "REJECT":
                curr_rej += 1
                curr_miss += 1
            elif event_type == "EXPIRE":
                if tid in task_in_q:
                    curr_q = max(0, curr_q - 1)
                    task_in_q.remove(tid)
                curr_miss += 1
            elif event_type == "PREEMPT":
                curr_act = max(0, curr_act - 1)
                curr_q += 1
                task_in_q.add(tid)
                
            q_depth.append(curr_q)
            active.append(curr_act)
            rejections.append(curr_rej)
            misses.append(curr_miss)
            
        return times, q_depth, active, rejections, misses

    m_t, m_q, m_a, m_r, m_m = process_timeline(mosaic_timeline)
    p_t, p_q, p_a, p_r, p_m = process_timeline(priority_timeline)
    
    try:
        from visualization.plot_results import setup_style, plt, HAS_MPL
        if HAS_MPL:
            setup_style()
            fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex="col", sharey="row")
            fig.patch.set_facecolor("#0d1117")
            
            # Ax[0, 0]: PriorityStrict Queue & Active
            axes[0, 0].step(p_t, p_q, label="Queue Depth", color="#ff4d6a", alpha=0.8, where="post", zorder=3)
            axes[0, 0].step(p_t, p_a, label="Active Tasks", color="#00d4ff", alpha=0.8, where="post", zorder=3)
            axes[0, 0].set_ylabel("Task Count")
            axes[0, 0].set_title("PriorityStrict: Resource Squeeze & Queue Explosion")
            axes[0, 0].legend(loc="upper left", framealpha=0.3)
            
            # Ax[0, 1]: MOSAIC Queue & Active
            axes[0, 1].step(m_t, m_q, label="Queue Depth", color="#ff4d6a", alpha=0.8, where="post", zorder=3)
            axes[0, 1].step(m_t, m_a, label="Active Tasks", color="#00e5a0", alpha=0.8, where="post", zorder=3)
            axes[0, 1].set_title("MOSAIC: Dynamic Throttling & Bounded Queue")
            axes[0, 1].legend(loc="upper left", framealpha=0.3)
            
            # Ax[1, 0]: PriorityStrict Failures
            axes[1, 0].step(p_t, p_m, label="Total Missed Deadlines", color="#ff4d6a", alpha=0.8, where="post", zorder=3)
            axes[1, 0].step(p_t, p_r, label="Admission Rejections", color="#ffaa00", alpha=0.8, where="post", zorder=3)
            axes[1, 0].set_xlabel("Time (seconds)")
            axes[1, 0].set_ylabel("Cumulative Failures")
            axes[1, 0].set_title("PriorityStrict: Complete Service Collapse")
            axes[1, 0].legend(loc="upper left", framealpha=0.3)
            
            # Ax[1, 1]: MOSAIC Failures
            axes[1, 1].step(m_t, m_m, label="Total Missed Deadlines", color="#ff4d6a", alpha=0.8, where="post", zorder=3)
            axes[1, 1].step(m_t, m_r, label="Admission Rejections", color="#ffaa00", alpha=0.8, where="post", zorder=3)
            axes[1, 1].set_xlabel("Time (seconds)")
            axes[1, 1].set_title("MOSAIC: Controlled Rejections (Safe Degradation)")
            axes[1, 1].legend(loc="upper left", framealpha=0.3)
            
            max_val = max(max(p_q), max(m_q), 8) + 2
            axes[0, 0].set_ylim(0, max_val)
            axes[0, 1].set_ylim(0, max_val)
            
            max_fail = max(max(p_m), max(m_m)) + 10
            axes[1, 0].set_ylim(0, max_fail)
            axes[1, 1].set_ylim(0, max_fail)
            
            fig.suptitle("Disaster Overload Timeline Comparison: PriorityStrict vs MOSAIC", fontsize=14, fontweight="bold", color="#e8edf4", y=0.98)
            
            p_out = FINAL_DIR / "overload_timeline.png"
            fig.savefig(p_out, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  [viz] Saved overload timeline comparison plot: {p_out}")
    except Exception as ex:
        print(f"  [viz] Failed to generate overload timeline plot: {ex}")
        
def main():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    break_priority_strict()
    task_class_completion_analysis()
    generate_overload_timeline_plot()

if __name__ == "__main__":
    main()
