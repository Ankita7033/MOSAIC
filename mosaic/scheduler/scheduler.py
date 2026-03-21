#!/usr/bin/env python3
"""
MOSAIC Scheduler -- Cross-Platform (Linux + Windows + macOS)
Uses Unix socket on Linux/macOS, TCP localhost on Windows.
"""
from __future__ import annotations
import os, sys, json, time, math, socket, signal, logging, argparse, threading
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

IS_WINDOWS = sys.platform == "win32"
IS_LINUX   = sys.platform == "linux"

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scheduler"))
sys.path.insert(0, str(_ROOT / "scheduler" / "core_algorithm"))

from core_algorithm import (
    WORKLOAD_CLASSES, CLASS_NAMES,
    compute_urgency, check_interference_admission,
    jains_fairness_index, detect_starvation,
    should_throttle, select_throttle_target,
    WorkloadClassifier,
)

DATA_DIR     = _ROOT / "data"
SOCK_PATH    = DATA_DIR / "mosaic.sock"
LOG_PATH     = DATA_DIR / "mosaic.log"
METRICS_PATH = DATA_DIR / "metrics.jsonl"
RAPL_PATH    = Path("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj")
TCP_HOST, TCP_PORT = "127.0.0.1", 47777
PORT_FILE    = DATA_DIR / "mosaic.port"

DEFAULT_POWER_CAP = 85.0
MAX_QUEUE_DEPTH   = 200
SCHEDULER_TICK_MS = 100
RAPL_INTERVAL_MS  = 500

DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(str(LOG_PATH), mode='a')]
)
log = logging.getLogger("mosaic")

@dataclass
class Task:
    task_id:str; workload_class:str; deadline_ms:int; priority:int
    mem_mb:int; gpu_required:bool; tier:int; cpu_shares:int
    submit_time:float=field(default_factory=time.monotonic)
    start_time:float=0.0; urgency:float=0.0

    def age_ms(self):               return (time.monotonic()-self.submit_time)*1000
    def deadline_remaining_ms(self):return max(0.0, self.deadline_ms-self.age_ms())
    def compute_urgency(self):
        u=compute_urgency(self.deadline_ms,self.age_ms(),self.tier,self.priority)
        self.urgency=u; return u

class RAPLReader:
    def __init__(self):
        self._path=RAPL_PATH; self._last_uj=None; self._last_ts=0.0; self._sim_w=48.0
        self._simulated=not(IS_LINUX and self._path.exists())
        if self._simulated:
            log.warning(f"[rapl] {'Windows' if IS_WINDOWS else 'non-Linux'} -- power simulated")
    def watts(self, n_running=0):
        if self._simulated:
            import random
            self._sim_w=45.0+n_running*5.2+random.gauss(0,2.0); return max(15.0,self._sim_w)
        try:
            uj=int(self._path.read_text(encoding="utf-8").strip()); now=time.monotonic()
            if self._last_uj is None: self._last_uj,self._last_ts=uj,now; return 0.0
            dt=now-self._last_ts
            if dt<0.05: return self._sim_w
            duj=uj-self._last_uj
            if duj<0: duj+=(1<<32)
            w=(duj/1e6)/dt; self._last_uj,self._last_ts,self._sim_w=uj,now,w; return w
        except: return self._sim_w

class CgroupManager:
    ROOT=Path("/sys/fs/cgroup/mosaic")
    def __init__(self):
        self._ok=IS_LINUX and self._try_init()
        if not self._ok: log.info(f"[cgroup] {'Windows' if IS_WINDOWS else 'non-Linux'} -- simulated")
    def _try_init(self):
        try: self.ROOT.mkdir(parents=True,exist_ok=True); return True
        except: return False
    def create(self,task_id,cpu_shares,mem_mb):
        if not self._ok: return True
        cg=self.ROOT/task_id
        try:
            cg.mkdir(exist_ok=True)
            (cg/"cpu.weight").write_text(str(min(10000,max(1,cpu_shares))))
            if mem_mb>0: (cg/"memory.max").write_text(str(mem_mb*1024*1024))
            return True
        except: return False
    def destroy(self,task_id):
        if not self._ok: return
        try:
            cg=self.ROOT/task_id
            if cg.exists(): cg.rmdir()
        except: pass
    def set_cpu_weight(self,task_id,weight):
        if not self._ok: return
        try:
            cg=self.ROOT/task_id
            if cg.exists(): (cg/"cpu.weight").write_text(str(max(1,weight)))
        except: pass

class MetricsLogger:
    def __init__(self):
        self._lock=threading.Lock(); self._latencies=deque(maxlen=2000)
        self._by_class={c:{"hits":0,"misses":0} for c in CLASS_NAMES}
    def log_event(self,event,task=None,power=0.0,extra=None):
        rec={"ts":round(time.time(),3),"event":event,"power_w":round(power,1)}
        if task: rec.update({"task_id":task.task_id,"class":task.workload_class,
                             "tier":task.tier,"deadline_ms":task.deadline_ms,
                             "urgency":round(task.urgency,3)})
        if extra: rec.update(extra)
        with self._lock:
            with open(METRICS_PATH,"a") as f: f.write(json.dumps(rec)+"\n")
    def record_completion(self,task,actual_ms):
        hit=actual_ms<=task.deadline_ms; slack=task.deadline_ms-actual_ms
        with self._lock:
            self._latencies.append(actual_ms)
            cls=task.workload_class
            if cls in self._by_class: self._by_class[cls]["hits" if hit else "misses"]+=1
        return hit,slack
    def percentile(self,p):
        with self._lock:
            if not self._latencies: return 0.0
            s=sorted(self._latencies); return s[max(0,int(math.ceil(p/100*len(s)))-1)]
    def fairness_index(self):
        with self._lock:
            rates=[d["hits"]/(d["hits"]+d["misses"]) for d in self._by_class.values() if d["hits"]+d["misses"]>0]
            return jains_fairness_index(rates) if rates else 1.0
    def class_hit_rates(self):
        with self._lock:
            return {cls:(d["hits"]/(d["hits"]+d["misses"])) if d["hits"]+d["misses"]>0 else 1.0
                    for cls,d in self._by_class.items()}

class MOSAICScheduler:
    def __init__(self,power_cap=DEFAULT_POWER_CAP):
        self._lock=threading.RLock(); self._running={}; self._queue=[]
        self._power_cap=power_cap; self._current_w=0.0; self._stop=threading.Event()
        self._rapl=RAPLReader(); self._cgroups=CgroupManager()
        self._mlog=MetricsLogger(); self._clf=WorkloadClassifier()
        self._admitted=self._queued=self._completed=self._hits=self._misses=self._throttle_events=0
        log.info(f"[mosaic] Ready  power_cap={power_cap}W  platform={sys.platform}")

    def submit(self,data):
        cls_name=data.get("class","")
        if cls_name not in WORKLOAD_CLASSES:
            r=self._clf.classify_from_metadata(data); cls_name=r.predicted_class
        wc=WORKLOAD_CLASSES[cls_name]
        task=Task(task_id=data.get("task_id",f"auto_{int(time.time()*1000)%99999:05d}"),
                  workload_class=cls_name,
                  deadline_ms=int(data.get("deadline_ms",wc.deadline_range[1])),
                  priority=int(data.get("priority",2)),mem_mb=int(data.get("mem_mb",wc.mem_mb)),
                  gpu_required=bool(data.get("gpu_required",wc.gpu_required)),
                  tier=wc.tier,cpu_shares=int(data.get("cpu_shares",wc.cpu_shares)))
        task.compute_urgency()
        with self._lock:
            dec=check_interference_admission(task.workload_class,task.deadline_ms,list(self._running.values()))
            if dec.admit: return self._admit(task)
            return self._enqueue(task,dec.reason)

    def complete(self,task_id,actual_ms,ipc=0,llc=0,bw=0,br=0):
        with self._lock:
            task=self._running.pop(task_id,None)
            if not task: return {"result":"error","reason":"task_not_found"}
            self._cgroups.destroy(task_id)
            hit,slack=self._mlog.record_completion(task,actual_ms)
            if hit: self._hits+=1
            else:   self._misses+=1
            self._completed+=1
            if ipc>0: self._clf.online_update(task.workload_class,ipc,llc,bw,br)
            # Update interference matrix with observed latency data
            self._update_interference_db(task, actual_ms)
            self._mlog.log_event("COMPLETE",task,power=self._current_w,
                                 extra={"actual_ms":round(actual_ms,1),"hit":hit,"slack_ms":round(slack,1)})
            log.info(f"[sched] COMPLETE {task_id} {'HIT' if hit else 'MISS'}")
            self._drain_queue()
            return {"result":"completed","task_id":task_id,"deadline_hit":hit,"slack_ms":round(slack,1)}

    def classify(self,ipc,llc,bw,br):
        r=self._clf.classify(ipc,llc,bw,br)
        return {"result":"classified","class":r.predicted_class,"confidence":r.confidence,
                "tier":r.tier,"distances":r.distances,"method":r.method}

    def status(self):
        with self._lock:
            total=self._hits+self._misses
            return {"result":"status",
                    "running":[{"task_id":t.task_id,"class":t.workload_class,"tier":t.tier,
                                "urgency":round(t.urgency,3),"deadline_ms":t.deadline_ms,
                                "deadline_remaining_ms":round(t.deadline_remaining_ms(),0),
                                "age_ms":round(t.age_ms(),0)} for t in self._running.values()],
                    "queued": [{"task_id":t.task_id,"class":t.workload_class,"tier":t.tier,
                                "urgency":round(t.urgency,3),
                                "deadline_remaining_ms":round(t.deadline_remaining_ms(),0)} for t in self._queue],
                    "metrics":{"admitted":self._admitted,"queued_total":self._queued,
                               "completed":self._completed,"hits":self._hits,"misses":self._misses,
                               "hit_rate":round(self._hits/total,4) if total else 1.0,
                               "p50_ms":round(self._mlog.percentile(50),1),
                               "p95_ms":round(self._mlog.percentile(95),1),
                               "p99_ms":round(self._mlog.percentile(99),1),
                               "power_watts":round(self._current_w,1),"power_cap":self._power_cap,
                               "throttle_events":self._throttle_events,
                               "queue_depth":len(self._queue),"running_count":len(self._running),
                               "fairness_index":round(self._mlog.fairness_index(),4),
                               "starvation_count":0,
                               "class_hit_rates":{k:round(v,3) for k,v in self._mlog.class_hit_rates().items()},
                               "platform":sys.platform}}

    def _admit(self,task):
        task.start_time=time.monotonic(); self._running[task.task_id]=task
        self._cgroups.create(task.task_id,task.cpu_shares,task.mem_mb)
        self._admitted+=1
        self._mlog.log_event("ADMIT",task,power=self._current_w)
        log.info(f"[sched] ADMIT  {task.task_id} {task.workload_class} tier={task.tier} urgency={task.urgency:.3f}")
        return {"result":"admitted","task_id":task.task_id,"class":task.workload_class,
                "tier":task.tier,"urgency":round(task.urgency,4)}

    def _enqueue(self,task,reason):
        if len(self._queue)>=MAX_QUEUE_DEPTH: return {"result":"rejected","task_id":task.task_id,"reason":"queue_full"}
        self._queue.append(task); self._queue.sort(key=lambda t:t.compute_urgency(),reverse=True)
        self._queued+=1
        self._mlog.log_event("QUEUE",task,power=self._current_w,extra={"reason":reason})
        log.info(f"[sched] QUEUE  {task.task_id} {task.workload_class} {reason}")
        return {"result":"queued","task_id":task.task_id,"reason":reason,"urgency":round(task.urgency,4)}

    def _drain_queue(self):
        for task in list(self._queue):
            if task.deadline_remaining_ms()<=0:
                self._misses+=1; self._mlog.log_event("EXPIRE",task); self._queue.remove(task); continue
            task.compute_urgency()
            dec=check_interference_admission(task.workload_class,task.deadline_ms,list(self._running.values()))
            if dec.admit:
                self._queue.remove(task); self._admit(task)
                if len(self._running)>=16: break
        self._queue.sort(key=lambda t:t.urgency,reverse=True)

    def _energy_loop(self):
        while not self._stop.is_set():
            self._current_w=self._rapl.watts(len(self._running))
            if should_throttle(self._current_w,self._power_cap):
                with self._lock:
                    target=select_throttle_target(list(self._running.values()))
                    if target:
                        ns=max(10,target.cpu_shares//2); self._cgroups.set_cpu_weight(target.task_id,ns)
                        target.cpu_shares=ns; self._throttle_events+=1
                        log.warning(f"[rapl] {self._current_w:.1f}W > cap -- throttle {target.task_id}")
            time.sleep(RAPL_INTERVAL_MS/1000.0)

    def _tick_loop(self):
        while not self._stop.is_set():
            with self._lock:
                for tid in detect_starvation(self._queue):
                    for t in self._queue:
                        if t.task_id==tid: t.urgency=math.inf
                for t in list(self._running.values())+self._queue: t.compute_urgency()
                self._queue.sort(key=lambda t:t.urgency,reverse=True)
                self._drain_queue()
            time.sleep(SCHEDULER_TICK_MS/1000.0)

    def _update_interference_db(self, completed_task, actual_ms):
        """
        After a task completes, update the interference matrix for all pairs
        that were colocated during its execution. This implements online learning
        for the interference model -- the DB gets more accurate over time.
        """
        try:
            import sys
            sys.path.insert(0, str(_ROOT / "profiler"))
            from build_db import get_conn, init_db, update_interference, DB_PATH
            conn = get_conn()
            init_db(conn)
            # For each task that was running alongside this one (approximately),
            # update their pairwise interference entry based on actual vs expected latency
            running_classes = [t.workload_class for t in self._running.values()]
            for rc in running_classes:
                if rc == completed_task.workload_class: continue
                # Observed overhead = actual_ms vs expected service time
                wc = WORKLOAD_CLASSES.get(completed_task.workload_class)
                if not wc: continue
                expected_ms = (wc.service_range[0] + wc.service_range[1]) / 2
                observed_overhead = max(0.0, actual_ms - expected_ms)
                if observed_overhead > 0:
                    # Convert to ipc_degradation estimate: overhead_fraction
                    ipc_deg_obs = min(0.95, observed_overhead / max(1.0, expected_ms))
                    update_interference(conn, rc, completed_task.workload_class,
                                        ipc_deg_obs, observed_overhead)
            conn.close()
        except Exception:
            pass  # Never crash the scheduler over a DB update

    def run_background(self):
        threading.Thread(target=self._energy_loop,daemon=True,name="rapl").start()
        threading.Thread(target=self._tick_loop,  daemon=True,name="tick").start()
        log.info("[mosaic] Background threads started")

    def stop(self):
        self._stop.set(); self._clf.save_centroids(); log.info("[mosaic] Stopped")


class SocketServer:
    def __init__(self,scheduler):
        self._s=scheduler

    def _handle(self,conn):
        buf=b""
        try:
            while True:
                chunk=conn.recv(4096)
                if not chunk: break
                buf+=chunk
                while b"\n" in buf:
                    line,buf=buf.split(b"\n",1); line=line.strip()
                    if not line: continue
                    try: msg=json.loads(line)
                    except json.JSONDecodeError as e:
                        conn.sendall((json.dumps({"result":"error","reason":str(e)})+"\n").encode()); continue
                    op=msg.get("op","")
                    if   op=="submit":   reply=self._s.submit(msg)
                    elif op=="complete": reply=self._s.complete(msg["task_id"],float(msg.get("actual_ms",0)),
                                                                float(msg.get("ipc",0)),float(msg.get("llc",0)),
                                                                float(msg.get("bw",0)),float(msg.get("br",0)))
                    elif op=="classify": reply=self._s.classify(float(msg.get("ipc",0)),float(msg.get("llc_miss_rate",0)),
                                                                float(msg.get("mem_bw_gbs",0)),float(msg.get("branch_miss_rate",0)))
                    elif op=="status":   reply=self._s.status()
                    elif op=="quit":     conn.sendall((json.dumps({"result":"bye"})+"\n").encode()); return
                    else: reply={"result":"error","reason":f"unknown op: {op}"}
                    conn.sendall((json.dumps(reply)+"\n").encode())
        except (ConnectionResetError,BrokenPipeError,OSError): pass
        finally:
            try: conn.close()
            except: pass

    def serve(self):
        DATA_DIR.mkdir(parents=True,exist_ok=True)
        srv=None; mode="tcp"
        if not IS_WINDOWS:
            try:
                if SOCK_PATH.exists(): SOCK_PATH.unlink()
                srv=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
                srv.bind(str(SOCK_PATH)); srv.listen(32); mode="unix"
                log.info(f"[server] Unix socket: {SOCK_PATH}"); PORT_FILE.write_text("unix")
            except OSError: srv=None
        if srv is None:
            srv=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            srv.bind((TCP_HOST,TCP_PORT)); srv.listen(32)
            log.info(f"[server] TCP socket: {TCP_HOST}:{TCP_PORT}"); PORT_FILE.write_text(str(TCP_PORT))
        try:
            while True:
                conn,_=srv.accept()
                threading.Thread(target=self._handle,args=(conn,),daemon=True).start()
        except KeyboardInterrupt: pass
        finally:
            srv.close()
            if mode=="unix" and SOCK_PATH.exists(): SOCK_PATH.unlink()
            if PORT_FILE.exists(): PORT_FILE.unlink()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--power-cap",type=float,default=DEFAULT_POWER_CAP)
    p.add_argument("--log-level",default="INFO",choices=["DEBUG","INFO","WARNING","ERROR"])
    args=p.parse_args(); logging.getLogger().setLevel(args.log_level)
    sched=MOSAICScheduler(args.power_cap); server=SocketServer(sched)
    def _shutdown(sig,_): sched.stop(); sys.exit(0)
    signal.signal(signal.SIGTERM,_shutdown); signal.signal(signal.SIGINT,_shutdown)
    sched.run_background(); server.serve()

if __name__=="__main__": main()