import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import json
import threading
import time

app = FastAPI()

# Global reference to telemetry data
# In a real deployment, this would be updated via Kafka/Redis or SSE
LATEST_TELEMETRY = {}

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>MOSAIC Cluster Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        body { font-family: monospace; background-color: #0d1117; color: #e8edf4; padding: 20px; }
        .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
        .card { background: #161e28; padding: 15px; border-radius: 8px; border: 1px solid #2a3a4a; }
        h1, h2 { color: #00e5a0; margin-top: 0; }
        .metric { font-size: 24px; font-weight: bold; margin: 10px 0; }
        .label { color: #8b9eb3; font-size: 14px; }
    </style>
</head>
<body>
    <h1>MOSAIC Distributed Cluster Monitor</h1>
    <div class="grid" id="node-container"></div>

    <script>
        async function fetchTelemetry() {
            try {
                const response = await fetch('/api/telemetry');
                const data = await response.json();
                
                if (data && data.nodes) {
                    const container = document.getElementById('node-container');
                    container.innerHTML = '';
                    
                    for (const [nodeId, metrics] of Object.entries(data.nodes)) {
                        const div = document.createElement('div');
                        div.className = 'card';
                        
                        const pmuColor = metrics.pmu_pressure > 1.5 ? '#ff4d6a' : '#00e5a0';
                        
                        div.innerHTML = `
                            <h2>${nodeId.toUpperCase()}</h2>
                            <div><span class="label">CPU Slots Active:</span> <span class="metric">${(metrics.cpu_util * 100).toFixed(0)}%</span></div>
                            <div><span class="label">Queue Depth:</span> <span class="metric">${metrics.queue_depth}</span></div>
                            <div><span class="label">PMU Pressure Score:</span> <span class="metric" style="color:${pmuColor}">${metrics.pmu_pressure.toFixed(2)}</span></div>
                            <div><span class="label">LLC Thrash Index:</span> <span class="metric">${metrics.llc_pressure.toFixed(2)}</span></div>
                            <hr style="border-color:#2a3a4a">
                            <div><span class="label">Migrations In/Out:</span> <span class="metric">${metrics.migrations_in} / ${metrics.migrations_out}</span></div>
                            <div><span class="label">Tasks Rejected:</span> <span class="metric">${metrics.rejections}</span></div>
                        `;
                        container.appendChild(div);
                    }
                }
            } catch (error) {
                console.error("Error fetching telemetry", error);
            }
        }
        
        setInterval(fetchTelemetry, 1000); // Poll every second
        fetchTelemetry();
    </script>
</body>
</html>
"""

@app.get("/")
async def get_dashboard():
    return HTMLResponse(content=HTML_CONTENT)

@app.get("/api/telemetry")
async def get_telemetry():
    return LATEST_TELEMETRY

def start_dashboard_server():
    """Starts the FastAPI server in a background thread."""
    def run():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
    
    server_thread = threading.Thread(target=run, daemon=True)
    server_thread.start()
    print("Dashboard running at http://localhost:8000")
    return server_thread

# For standalone testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
