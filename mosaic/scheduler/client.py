#!/usr/bin/env python3
"""MOSAIC Client -- cross-platform (Unix socket on Linux/macOS, TCP on Windows)"""
from __future__ import annotations
import json, socket, sys, time
from pathlib import Path

IS_WINDOWS   = sys.platform == "win32"
DEFAULT_SOCK = Path(__file__).parent.parent / "data" / "mosaic.sock"
PORT_FILE    = Path(__file__).parent.parent / "data" / "mosaic.port"
TCP_HOST, TCP_PORT = "127.0.0.1", 47777

class MOSAICClient:
    def __init__(self, timeout=5.0):
        self._timeout=timeout; self._conn=None; self._buf=b""

    def _detect_mode(self):
        """Returns ('unix', path) or ('tcp', port)"""
        if PORT_FILE.exists():
            val = PORT_FILE.read_text(encoding="utf-8").strip()
            if val == "unix" and DEFAULT_SOCK.exists():
                return "unix", str(DEFAULT_SOCK)
            try: return "tcp", int(val)
            except: pass
        if not IS_WINDOWS and DEFAULT_SOCK.exists():
            return "unix", str(DEFAULT_SOCK)
        return "tcp", TCP_PORT

    def connect(self):
        mode, addr = self._detect_mode()
        if mode == "unix":
            self._conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._conn.settimeout(self._timeout)
            try:
                self._conn.connect(addr)
            except (FileNotFoundError, ConnectionRefusedError):
                raise ConnectionRefusedError(
                    "MOSAIC not running. Start it:\n"
                    "  Windows: python scheduler\\scheduler.py\n"
                    "  Linux:   python3 scheduler/scheduler.py"
                )
        else:
            self._conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._conn.settimeout(self._timeout)
            try:
                self._conn.connect((TCP_HOST, addr))
            except ConnectionRefusedError:
                raise ConnectionRefusedError(
                    f"MOSAIC not running on {TCP_HOST}:{addr}. Start it:\n"
                    f"  Windows: python scheduler\\scheduler.py\n"
                    f"  Linux:   python3 scheduler/scheduler.py"
                )
        return self

    def __enter__(self): return self.connect()
    def __exit__(self, *_): self.close()

    def close(self):
        if self._conn:
            try: self._send({"op":"quit"})
            except: pass
            self._conn.close(); self._conn=None

    def _send(self, msg):
        if not self._conn: self.connect()
        try:
            self._conn.sendall((json.dumps(msg)+"\n").encode())
            return self._recv()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            try: self._conn.close()
            except: pass
            self._conn = None; self._buf = b""
            import time; time.sleep(0.5)
            self.connect()
            self._conn.sendall((json.dumps(msg)+"\n").encode())
            return self._recv()

    def _recv(self):
        while b"\n" not in self._buf:
            chunk=self._conn.recv(8192)
            if not chunk: raise ConnectionError("Scheduler closed connection")
            self._buf+=chunk
        line,self._buf=self._buf.split(b"\n",1)
        return json.loads(line.decode())

    def submit(self, task_id, workload_class, deadline_ms,
               priority=2, cpu_shares=512, mem_mb=512, gpu_required=False):
        return self._send({"op":"submit","task_id":task_id,"class":workload_class,
                           "deadline_ms":deadline_ms,"priority":priority,
                           "cpu_shares":cpu_shares,"mem_mb":mem_mb,"gpu_required":gpu_required})

    def complete(self, task_id, actual_ms, ipc=0.0, llc=0.0, bw=0.0, br=0.0):
        return self._send({"op":"complete","task_id":task_id,"actual_ms":actual_ms,
                           "ipc":ipc,"llc":llc,"bw":bw,"br":br})

    def classify(self, ipc, llc_miss_rate, mem_bw_gbs, branch_miss_rate):
        return self._send({"op":"classify","ipc":ipc,"llc_miss_rate":llc_miss_rate,
                           "mem_bw_gbs":mem_bw_gbs,"branch_miss_rate":branch_miss_rate})

    def status(self): return self._send({"op":"status"})

if __name__=="__main__":
    with MOSAICClient() as c:
        s=c.status(); m=s.get("metrics",{})
        print(f"\nMOSAIC Status  (platform={m.get('platform','?')})")
        print(f"  Running : {m.get('running_count',0)}  Queued: {m.get('queue_depth',0)}")
        print(f"  Hit rate: {m.get('hit_rate',0):.1%}  ({m.get('hits',0)} hits / {m.get('misses',0)} misses)")
        print(f"  P95/P99 : {m.get('p95_ms',0):.0f}ms / {m.get('p99_ms',0):.0f}ms")
        print(f"  Power   : {m.get('power_watts',0):.1f}W / {m.get('power_cap',85)}W")
        print(f"  Fairness: JFI={m.get('fairness_index',0):.4f}")