#!/usr/bin/env python3
"""
MOSAIC Fingerprint Database Builder
Initialises SQLite DB with interference matrix and workload fingerprints.
"""
from __future__ import annotations
import sqlite3, json, argparse, sys, math, time
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "fingerprints.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS fingerprints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workload_class TEXT NOT NULL,
    ipc REAL, llc_miss_rate REAL, mem_bw_gb_s REAL, branch_miss_rate REAL,
    captured_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS interference_matrix (
    class_a TEXT NOT NULL, class_b TEXT NOT NULL,
    ipc_degradation REAL NOT NULL, lat_overhead_ms REAL NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5, sample_count INTEGER DEFAULT 1,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (class_a, class_b)
);
CREATE TABLE IF NOT EXISTS scheduler_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT, workload_class TEXT, task_id TEXT,
    deadline_ms INTEGER, actual_ms INTEGER, urgency_score REAL,
    power_watts REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

DEFAULT_MATRIX = [
    ("inference_critical","dispatch_api",    0.38,22.0),
    ("inference_critical","sensor_fusion",   0.25,14.0),
    ("inference_critical","analytics_batch", 0.12, 6.0),
    ("inference_critical","model_update",    0.08, 3.0),
    ("inference_critical","log_archive",     0.04, 1.0),
    ("dispatch_api","inference_critical",    0.05, 2.0),
    ("dispatch_api","sensor_fusion",         0.10, 4.0),
    ("dispatch_api","analytics_batch",       0.06, 2.0),
    ("dispatch_api","model_update",          0.04, 1.5),
    ("dispatch_api","log_archive",           0.02, 0.5),
    ("sensor_fusion","inference_critical",   0.18, 9.0),
    ("sensor_fusion","dispatch_api",         0.14, 6.0),
    ("sensor_fusion","analytics_batch",      0.10, 4.0),
    ("sensor_fusion","model_update",         0.12, 5.0),
    ("sensor_fusion","log_archive",          0.03, 1.0),
    ("analytics_batch","inference_critical", 0.20,11.0),
    ("analytics_batch","dispatch_api",       0.16, 7.0),
    ("analytics_batch","sensor_fusion",      0.14, 6.0),
    ("analytics_batch","model_update",       0.18, 8.0),
    ("analytics_batch","log_archive",        0.05, 1.5),
    ("model_update","inference_critical",    0.22,12.0),
    ("model_update","dispatch_api",          0.18, 9.0),
    ("model_update","sensor_fusion",         0.16, 7.0),
    ("model_update","analytics_batch",       0.14, 6.0),
    ("model_update","log_archive",           0.06, 2.0),
    ("log_archive","inference_critical",     0.03, 1.0),
    ("log_archive","dispatch_api",           0.02, 0.5),
    ("log_archive","sensor_fusion",          0.02, 0.5),
    ("log_archive","analytics_batch",        0.03, 1.0),
    ("log_archive","model_update",           0.02, 0.5),
]

CLASSES = ["inference_critical","dispatch_api","sensor_fusion",
           "analytics_batch","model_update","log_archive"]

def get_conn(path=None):
    p = path or DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    return conn

def init_db(conn):
    conn.executescript(SCHEMA)
    cur = conn.execute("SELECT COUNT(*) FROM interference_matrix")
    if cur.fetchone()[0] == 0:
        for row in DEFAULT_MATRIX:
            conn.execute(
                "INSERT OR IGNORE INTO interference_matrix "
                "(class_a,class_b,ipc_degradation,lat_overhead_ms,confidence,sample_count) "
                "VALUES (?,?,?,?,0.5,1)", row)
        for cls in CLASSES:
            conn.execute(
                "INSERT OR IGNORE INTO interference_matrix "
                "(class_a,class_b,ipc_degradation,lat_overhead_ms,confidence,sample_count) "
                "VALUES (?,?,0.0,0.0,1.0,1)", (cls,cls))
        conn.commit()

def update_interference(conn, class_a, class_b, observed_ipc, observed_lat):
    alpha = 0.3
    row = conn.execute(
        "SELECT ipc_degradation,lat_overhead_ms,sample_count FROM interference_matrix "
        "WHERE class_a=? AND class_b=?", (class_a,class_b)).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO interference_matrix "
            "(class_a,class_b,ipc_degradation,lat_overhead_ms,confidence,sample_count) "
            "VALUES (?,?,?,?,0.3,1)", (class_a,class_b,observed_ipc,observed_lat))
    else:
        n = row["sample_count"] + 1
        new_ipc = (1-alpha)*row["ipc_degradation"] + alpha*observed_ipc
        new_lat = (1-alpha)*row["lat_overhead_ms"]  + alpha*observed_lat
        conf    = min(1.0, 1.0 - math.exp(-n/10.0))
        conn.execute(
            "UPDATE interference_matrix SET ipc_degradation=?,lat_overhead_ms=?,"
            "confidence=?,sample_count=?,updated_at=CURRENT_TIMESTAMP "
            "WHERE class_a=? AND class_b=?",
            (new_ipc,new_lat,conf,n,class_a,class_b))
    conn.commit()

def show_matrix(conn):
    classes = [r[0] for r in conn.execute(
        "SELECT DISTINCT class_a FROM interference_matrix ORDER BY class_a").fetchall()]
    if not classes: print("Matrix empty."); return
    w = 16
    print("\nInterference Matrix (IPC degradation -- how much row degrades column)")
    print("-"*(w*(len(classes)+1)+2))
    print(f"{'':>{w}}"+"".join(f"{c:>{w}}" for c in classes))
    print("-"*(w*(len(classes)+1)+2))
    for ca in classes:
        row = f"{ca:>{w}}"
        for cb in classes:
            r = conn.execute(
                "SELECT ipc_degradation FROM interference_matrix WHERE class_a=? AND class_b=?",
                (ca,cb)).fetchone()
            row += f"{(r['ipc_degradation'] if r else 0):>{w}.3f}"
        print(row)

def main():
    parser = argparse.ArgumentParser(description="MOSAIC Fingerprint DB")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--show",   action="store_true")
    parser.add_argument("--ingest", metavar="JSON")
    parser.add_argument("--update-interference", nargs=4,
                        metavar=("A","B","IPC","LAT"))
    args = parser.parse_args()
    conn = get_conn()
    init_db(conn)
    if args.matrix:
        show_matrix(conn)
    elif args.show:
        rows = conn.execute(
            "SELECT workload_class,ipc,llc_miss_rate,mem_bw_gb_s,captured_at "
            "FROM fingerprints ORDER BY captured_at DESC LIMIT 50").fetchall()
        print(f"\n{'Class':<22} {'IPC':>8} {'LLC_miss':>10} {'MemBW':>8}")
        for r in rows:
            print(f"{r['workload_class']:<22} {r['ipc']:>8.3f} "
                  f"{r['llc_miss_rate']:>10.4f} {r['mem_bw_gb_s']:>8.2f}")
    elif args.ingest:
        with open(args.ingest, encoding="utf-8") as f:
            data = json.load(f)
        conn.execute(
            "INSERT INTO fingerprints (workload_class,ipc,llc_miss_rate,mem_bw_gb_s,branch_miss_rate) "
            "VALUES (?,?,?,?,?)",
            (data.get("workload_class","unknown"), data.get("ipc",0),
             data.get("llc_miss_rate",0), data.get("mem_bw_gb_s",0),
             data.get("branch_miss_rate",0)))
        conn.commit()
        print(f"Ingested: {data.get('workload_class')}")
    elif args.update_interference:
        ca,cb,ipc,lat = args.update_interference
        update_interference(conn, ca, cb, float(ipc), float(lat))
        print(f"Updated ({ca} → {cb})")
    else:
        parser.print_help()
    conn.close()

if __name__ == "__main__":
    main()
