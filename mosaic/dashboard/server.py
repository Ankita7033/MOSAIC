#!/usr/bin/env python3
"""
MOSAIC Live Dashboard Server
Serves all data from the running scheduler and SQLite DB.

Endpoints:
  GET /              → dashboard HTML
  GET /stream        → SSE: all live data every 500ms (metrics + matrix + tasks + events)
  GET /interference  → JSON: full 6x6 interference matrix with confidence
  GET /metrics       → SSE: metrics only
  GET /status        → JSON: one-shot full status
  GET /health        → JSON: alive check
  POST /submit       → submit a task to the scheduler
"""
from __future__ import annotations

import sys, os, json, time, math, socket, threading, argparse, sqlite3
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

_ROOT        = Path(__file__).parent.parent
SOCK_PATH    = _ROOT / "data" / "mosaic.sock"
PORT_FILE    = _ROOT / "data" / "mosaic.port"
METRICS_PATH = _ROOT / "data" / "metrics.jsonl"
DB_PATH      = _ROOT / "data" / "fingerprints.db"

TCP_HOST, TCP_PORT = "127.0.0.1", 47777
IS_WINDOWS = sys.platform == "win32"

sys.path.insert(0, str(_ROOT / "scheduler" / "core_algorithm"))
from workload_taxonomy import WORKLOAD_CLASSES, CLASS_NAMES, INTERFERENCE_MATRIX as STATIC_MATRIX

# ── Interference matrix reader ────────────────────────────────────────────────

def read_interference_matrix() -> dict:
    """
    Read the live interference matrix from SQLite DB.
    Falls back to static hardcoded matrix if DB unavailable.
    Returns nested dict: matrix[class_a][class_b] = {ipc, lat, confidence, n}
    """
    matrix = {}
    try:
        if DB_PATH.exists():
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT class_a, class_b, ipc_degradation, lat_overhead_ms, "
                "       confidence, sample_count, updated_at "
                "FROM interference_matrix ORDER BY class_a, class_b"
            ).fetchall()
            conn.close()
            for r in rows:
                ca, cb = r["class_a"], r["class_b"]
                if ca not in matrix: matrix[ca] = {}
                matrix[ca][cb] = {
                    "ipc_degradation": round(r["ipc_degradation"], 4),
                    "lat_overhead_ms": round(r["lat_overhead_ms"], 2),
                    "confidence":      round(r["confidence"], 3),
                    "sample_count":    r["sample_count"],
                    "updated_at":      r["updated_at"] or "",
                    "live": True,
                }
            if matrix:
                return matrix
    except Exception:
        pass

    # Fallback: static matrix from taxonomy
    for ca, victims in STATIC_MATRIX.items():
        matrix[ca] = {}
        for cb, (ipc, lat) in victims.items():
            matrix[ca][cb] = {
                "ipc_degradation": ipc,
                "lat_overhead_ms": lat,
                "confidence": 0.5,
                "sample_count": 1,
                "updated_at": "",
                "live": False,
            }
    return matrix


# ── Scheduler bridge ──────────────────────────────────────────────────────────

class SchedulerBridge:
    def __init__(self):
        self._lock  = threading.Lock()
        self._cache = {}
        self._events_cache = []

    def _detect(self):
        if PORT_FILE.exists():
            val = PORT_FILE.read_text(encoding="utf-8").strip()
            if val == "unix" and SOCK_PATH.exists():
                return "unix", str(SOCK_PATH)
            try: return "tcp", int(val)
            except: pass
        if not IS_WINDOWS and SOCK_PATH.exists():
            return "unix", str(SOCK_PATH)
        return "tcp", TCP_PORT

    def _query(self, msg: dict) -> dict:
        try:
            mode, addr = self._detect()
            if mode == "unix":
                conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                conn.settimeout(2.0)
                conn.connect(addr)
            else:
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.settimeout(2.0)
                conn.connect((TCP_HOST, addr))
            conn.sendall((json.dumps(msg) + "\n").encode())
            buf = b""
            while b"\n" not in buf:
                chunk = conn.recv(4096)
                if not chunk: break
                buf += chunk
            conn.close()
            return json.loads(buf.split(b"\n")[0].decode())
        except Exception:
            return {}

    def alive(self) -> bool:
        return SOCK_PATH.exists() or PORT_FILE.exists()

    def fetch_status(self) -> dict:
        data = self._query({"op": "status"})
        if data:
            with self._lock: self._cache = data
        return self._cache

    def submit(self, task: dict) -> dict:
        return self._query({**task, "op": "submit"})

    def read_events(self, n: int = 25) -> list:
        events = []
        if not METRICS_PATH.exists(): return events
        try:
            lines = METRICS_PATH.read_text(encoding="utf-8").strip().split("\n")
            for line in reversed(lines[-300:]):
                try:
                    events.append(json.loads(line))
                    if len(events) >= n: break
                except: pass
        except: pass
        return events


bridge = SchedulerBridge()


# ── Full state builder ─────────────────────────────────────────────────────────

def build_full_state() -> dict:
    """
    Builds the complete state object sent to the dashboard on every SSE tick.
    Includes: metrics, running tasks, queued tasks, interference matrix, events.
    """
    status = bridge.fetch_status()
    matrix = read_interference_matrix()
    events = bridge.read_events(30)

    metrics  = status.get("metrics", {})
    running  = status.get("running", [])
    queued   = status.get("queued",  [])

    # Compute per-class urgency for running tasks
    for task in running:
        dl    = task.get("deadline_ms", 1000)
        rem   = task.get("deadline_remaining_ms", dl)
        tier  = task.get("tier", 2)
        tier_w = {1: 4.0, 2: 3.0, 3: 1.5, 4: 0.5}.get(tier, 1.0)
        frac  = rem / max(1.0, dl)
        task["urgency_live"] = round(tier_w / max(0.001, frac), 3)

    return {
        "ts":       round(time.time(), 3),
        "alive":    bridge.alive(),
        "metrics":  metrics,
        "running":  running,
        "queued":   queued,
        "matrix":   matrix,
        "events":   events,
        "classes":  CLASS_NAMES,
    }


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    def log_message(self, *_): pass  # suppress access log

    def _send(self, code, ct, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, gen_fn, interval=0.5):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            while True:
                data = gen_fn()
                self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()
                time.sleep(interval)
        except (BrokenPipeError, ConnectionResetError, OSError): pass

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"

        if path == "/":
            p = Path(__file__).parent / "dashboard.html"
            if p.exists():
                self._send(200, "text/html; charset=utf-8", p.read_bytes())
            else:
                self._send(404, "text/plain", b"dashboard.html not found")

        elif path == "/stream":
            # Main SSE stream — everything in one feed
            self._sse(build_full_state, interval=0.5)

        elif path == "/metrics":
            # Metrics-only SSE
            self._sse(lambda: bridge.fetch_status().get("metrics", {}), 0.5)

        elif path == "/interference":
            # One-shot JSON of full interference matrix
            matrix = read_interference_matrix()
            body = json.dumps({
                "matrix": matrix,
                "classes": CLASS_NAMES,
                "ts": round(time.time(), 3),
                "source": "sqlite" if DB_PATH.exists() else "static",
            }).encode()
            self._send(200, "application/json", body)

        elif path == "/status":
            data = bridge.fetch_status()
            self._send(200, "application/json", json.dumps(data).encode())

        elif path == "/health":
            alive = bridge.alive()
            body  = json.dumps({"alive": alive, "ts": round(time.time(), 3)})
            self._send(200, "application/json", body.encode())

        elif path == "/events":
            self._sse(lambda: {"events": bridge.read_events(25)}, 0.4)

        else:
            self._send(404, "text/plain", b"Not found")

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/submit":
            length = int(self.headers.get("Content-Length", 0))
            try:
                task   = json.loads(self.rfile.read(length))
                result = bridge.submit(task)
                self._send(200, "application/json", json.dumps(result).encode())
            except Exception as e:
                self._send(400, "application/json",
                           json.dumps({"error": str(e)}).encode())
        else:
            self._send(404, "text/plain", b"Not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ── Dashboard HTML generator ──────────────────────────────────────────────────

def _generate_live_dashboard(path: Path):
    """
    Inject SSE client into the static dashboard so it pulls all data
    from the backend — including the live interference matrix.
    """
    static = Path(__file__).parent / "dashboard.html"
    if not static.exists():
        path.write_text("<h1>dashboard.html missing</h1>", encoding="utf-8")
        return

    html = static.read_text(encoding="utf-8")

    sse_inject = r"""
<script>
(function() {
  var BASE = '';
  var stream = null;
  var connected = false;

  function connect() {
    if (stream) { try { stream.close(); } catch(e) {} }
    stream = new EventSource(BASE + '/stream');

    stream.onopen = function() {
      connected = true;
      var pill = document.querySelector('.live-pill');
      if (pill) pill.innerHTML = '<span class="pulse"></span>Live (backend)';
    };

    stream.onmessage = function(e) {
      try {
        var d = JSON.parse(e.data);
        if (d.metrics)    applyMetrics(d.metrics);
        if (d.running)    applyRunning(d.running, d.queued || []);
        if (d.events)     applyEvents(d.events);
        if (d.matrix)     applyMatrix(d.matrix, d.classes || []);
      } catch(err) { console.error('SSE parse error:', err); }
    };

    stream.onerror = function() {
      connected = false;
      var pill = document.querySelector('.live-pill');
      if (pill) pill.innerHTML = '<span class="pulse" style="background:#ffaa20"></span>Reconnecting...';
      setTimeout(connect, 3000);
    };
  }

  // ── Apply metrics to KPI cards ──────────────────────────────────────────
  function applyMetrics(m) {
    if (!m || !Object.keys(m).length) return;
    var total = (m.hits || 0) + (m.misses || 0);
    var hr = total > 0 ? m.hits / total : 0;

    setText('kHit',      (hr * 100).toFixed(1) + '%');
    setText('kHitSub',   (m.hits||0) + ' hits / ' + (m.misses||0) + ' misses');
    setText('kP99',      Math.round(m.p99_ms || 0) + 'ms');
    setText('kP99Sub',   'P95: ' + Math.round(m.p95_ms || 0) + 'ms');
    setText('kPow',      (m.power_watts || 0).toFixed(1) + 'W');
    setText('kJFI',      (m.fairness_index || 0).toFixed(4));
    var starveRate = total > 0 ? ((m.starvation_count||0) / total * 100) : 0;
    setText('kStarve',   starveRate.toFixed(1) + '%');
    setText('kStarveSub',(m.starvation_count||0) + ' starved tasks');
    setText('kPowSub',   'Cap: ' + (m.power_cap || 85) + ' W');

    // Update badge colours
    var bdgH = document.getElementById('bdgHit');
    if (bdgH) {
      if (hr >= 0.88) { bdgH.className='badge bg'; bdgH.textContent='HEALTHY'; }
      else if (hr >= 0.72) { bdgH.className='badge ba'; bdgH.textContent='DEGRADED'; }
      else { bdgH.className='badge br'; bdgH.textContent='CRITICAL'; }
    }
    var bdgP = document.getElementById('bdgPow');
    if (bdgP) {
      var pw = m.power_watts || 0, cap = m.power_cap || 85;
      if (pw > cap)          { bdgP.className='badge br'; bdgP.textContent='THROTTLING'; }
      else if (pw > cap*0.9) { bdgP.className='badge ba'; bdgP.textContent='NEAR CAP'; }
      else                   { bdgP.className='badge bg'; bdgP.textContent='NORMAL'; }
    }

    // Push metrics into chart histories
    if (window._mosaic_push) window._mosaic_push(m);
  }

  // ── Apply running/queued tasks to sidebar ───────────────────────────────
  function applyRunning(running, queued) {
    var el = document.getElementById('taskList');
    if (!el) return;
    if (!running.length) {
      el.innerHTML = '<div style="color:var(--dim);font-size:10px;padding:6px">No tasks running</div>';
      return;
    }
    var COLORS = {
      inference_critical:'#ff4060', dispatch_api:'#00e59a',
      sensor_fusion:'#00d4ff',      analytics_batch:'#ffa520',
      model_update:'#a78bfa',       log_archive:'#6a8098'
    };
    el.innerHTML = running.map(function(t) {
      var col = COLORS[t.class] || '#888';
      var tier = t.tier || 2;
      var urg  = (t.urgency || t.urgency_live || 0).toFixed(2);
      var rem  = t.deadline_remaining_ms || 0;
      var dl   = t.deadline_ms || 1000;
      var prog = Math.min(100, Math.round((1 - rem/dl) * 100));
      return '<div class="task-row" style="--cls:' + col + '">' +
        '<div class="task-top"><span class="task-id">' + t.task_id + ' T' + tier + '</span>' +
        '<span class="task-urg">u=' + urg + '</span></div>' +
        '<div class="task-cls">' + (t.class||'').replace(/_/g,' ') + '</div>' +
        '<div class="task-bar-bg"><div class="task-bar-fill" style="width:' + prog + '%;background:' + col + '"></div></div>' +
        '</div>';
    }).join('');
  }

  // ── Apply events to event log ────────────────────────────────────────────
  function applyEvents(events) {
    var el = document.getElementById('elog');
    if (!el || !events.length) return;
    el.innerHTML = events.map(function(e) {
      var ts = e.ts ? new Date(e.ts * 1000).toLocaleTimeString() : '--:--:--';
      var ev = e.event || '?';
      var msg = (e.task_id || '') + ' [' + ((e.class||'').replace(/_/g,' ')) + ']';
      if (e.actual_ms) msg += ' ' + Math.round(e.actual_ms) + 'ms ' + (e.hit ? 'HIT' : 'MISS');
      return '<div class="erow"><span class="ets">' + ts + '</span>' +
             '<span class="etype ' + ev + '">' + ev + '</span>' +
             '<span class="emsg">' + msg + '</span></div>';
    }).join('');
  }

  // ── Apply LIVE interference matrix ──────────────────────────────────────
  function applyMatrix(matrix, classes) {
    var container = document.getElementById('matWrap');
    if (!container || !matrix || !classes.length) return;

    var SIZE = 36, LW = 100;
    var html = '<div style="display:inline-block;margin-top:6px">';

    // Header row
    html += '<div style="display:grid;grid-template-columns:' + LW + 'px ' + classes.map(function(){return SIZE+'px';}).join(' ') + ';gap:2px;margin-bottom:2px">';
    html += '<div></div>';
    classes.forEach(function(c) {
      html += '<div style="font-size:7px;color:var(--muted);text-align:center;transform:rotate(-35deg);transform-origin:50% 80%;height:46px;display:flex;align-items:flex-end;justify-content:center;padding-bottom:2px;white-space:nowrap">' +
              c.replace(/_/g,'<br>') + '</div>';
    });
    html += '</div>';

    // Data rows
    classes.forEach(function(ca) {
      html += '<div style="display:grid;grid-template-columns:' + LW + 'px ' + classes.map(function(){return SIZE+'px';}).join(' ') + ';gap:2px;margin-bottom:2px">';
      html += '<div style="font-size:8px;color:var(--muted);display:flex;align-items:center;justify-content:flex-end;padding-right:6px">' + ca.replace(/_/g,' ') + '</div>';

      classes.forEach(function(cb) {
        var cell = (matrix[ca] && matrix[ca][cb]) ? matrix[ca][cb] : {ipc_degradation:0,lat_overhead_ms:0,confidence:0.5,sample_count:1};
        var v    = cell.ipc_degradation;
        var pct  = Math.round(v * 100);
        var conf = cell.confidence || 0.5;
        var n    = cell.sample_count || 1;
        var live = cell.live !== false;

        var bg, fc;
        if (v === 0)      { bg = 'rgba(255,255,255,0.04)'; fc = '#2e4055'; }
        else if (v < 0.10){ bg = 'rgba(0,229,154,' + (0.15+v) + ')'; fc = '#00e59a'; }
        else if (v < 0.20){ bg = 'rgba(255,165,32,' + (0.15+v) + ')'; fc = '#ffa520'; }
        else if (v < 0.30){ bg = 'rgba(255,122,92,' + (0.2+v)  + ')'; fc = '#ff7a5c'; }
        else              { bg = 'rgba(255,64,96,'  + (0.25+v) + ')'; fc = '#ff4060'; }

        var border = v >= 0.35 ? '2px solid #ff4060' : '1px solid rgba(255,255,255,0.05)';
        // Confidence indicator: dimmer if low confidence / few samples
        var opacity = 0.5 + conf * 0.5;

        var tooltip = ca.replace(/_/g,' ') + ' -> ' + cb.replace(/_/g,' ') + ': ' +
                      pct + '% IPC degradation, +' + cell.lat_overhead_ms + 'ms' +
                      ' | conf=' + (conf*100).toFixed(0) + '% n=' + n +
                      (live ? ' (live DB)' : ' (static)');

        html += '<div title="' + tooltip + '" style="width:' + SIZE + 'px;height:' + SIZE + 'px;' +
                'background:' + bg + ';color:' + fc + ';border:' + border + ';' +
                'border-radius:3px;display:flex;align-items:center;justify-content:center;' +
                'font-size:8px;font-weight:700;cursor:default;opacity:' + opacity + ';' +
                'transition:transform .15s,opacity .3s" ' +
                'onmouseenter="this.style.transform=\'scale(1.2)\';this.style.opacity=\'1\'" ' +
                'onmouseleave="this.style.transform=\'scale(1)\';this.style.opacity=\'' + opacity + '\'">' +
                (v === 0 ? '.' : pct) + '</div>';
      });
      html += '</div>';
    });

    // Legend with confidence explanation
    html += '<div style="display:flex;gap:14px;margin-top:10px;font-size:8px;color:var(--muted);flex-wrap:wrap">';
    [['Low <10%','#00e59a'],['Medium 10-20%','#ffa520'],['High 20-30%','#ff7a5c'],['Critical >30%','#ff4060']].forEach(function(pair) {
      html += '<span style="display:flex;align-items:center;gap:4px"><span style="width:9px;height:9px;border-radius:2px;background:' + pair[1] + ';opacity:.8;display:inline-block"></span>' + pair[0] + '</span>';
    });
    html += '</div>';
    html += '<div style="margin-top:6px;font-size:8px;color:var(--dim)">Opacity = confidence level. Hover for details. Updates live from SQLite DB.</div>';
    html += '</div>';

    container.innerHTML = html;
  }

  // ── Utility ─────────────────────────────────────────────────────────────
  function setText(id, val) {
    var el = document.getElementById(id);
    if (el) el.textContent = val;
  }

  // Start connection
  connect();

  // Expose push function for chart updates (called by main dashboard JS)
  window._mosaic_backend_connected = true;

})();
</script>
"""

    html = html.replace("</body>", sse_inject + "\n</body>")
    path.write_text(html, encoding="utf-8")
    pass  # dashboard.html served directly


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7777)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    server   = HTTPServer((args.host, args.port), Handler)
    is_alive = bridge.alive()

    print(f"\n  MOSAIC Live Dashboard")
    print(f"  {'='*42}")
    print(f"  URL        : http://localhost:{args.port}")
    sched_status = 'CONNECTED' if is_alive else 'not running (run: mosaic start)'
    print(f"  Scheduler  : {sched_status}")
    print(f"  Stream     : http://localhost:{args.port}/stream  (SSE - all live data)")
    print(f"  Matrix     : http://localhost:{args.port}/interference  (JSON)")
    print(f"  Status     : http://localhost:{args.port}/status  (JSON)")
    print(f"  {'='*42}")
    print(f"  Matrix source: {'SQLite DB (live)' if DB_PATH.exists() else 'static fallback'}")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
