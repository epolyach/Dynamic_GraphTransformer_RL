#!/usr/bin/env python3

"""
GPU Cluster Web Monitor
Provides a web interface to monitor GPU usage across multiple servers
"""

import subprocess
import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template_string, jsonify
import argparse

app = Flask(__name__)

# Configuration
SERVERS = ["gpu1.sedan.pro", "gpu2.sedan.pro", "gpu3.sedan.pro"]
UPDATE_INTERVAL = 10  # seconds
SSH_TIMEOUT = 5

# Global state
gpu_status = {}
last_update = None
lock = threading.Lock()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>GPU Cluster Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .timestamp {
            color: rgba(255,255,255,0.9);
            text-align: center;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .server-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .server-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .server-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.25);
        }
        .server-name {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e0e0e0;
        }
        .gpu-item {
            margin: 15px 0;
            padding: 15px;
            border-radius: 10px;
            background: #f5f5f5;
        }
        .gpu-free {
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            animation: pulse 2s infinite;
        }
        .gpu-busy {
            background: linear-gradient(135deg, #fc5c7d 0%, #f093fb 100%);
        }
        .gpu-partial {
            background: linear-gradient(135deg, #fddb92 0%, #d1fdff 100%);
        }
        .offline {
            background: #e0e0e0;
            opacity: 0.6;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(132, 250, 176, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(132, 250, 176, 0); }
            100% { box-shadow: 0 0 0 0 rgba(132, 250, 176, 0); }
        }
        .gpu-stats {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
            font-size: 0.9em;
        }
        .stat {
            padding: 5px 10px;
            background: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .user-info {
            margin-top: 10px;
            padding: 8px;
            background: white;
            border-radius: 5px;
            font-weight: 500;
            color: #333;
        }
        .alert-banner {
            background: linear-gradient(90deg, #00C851 0%, #00ff00 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
            font-size: 1.3em;
            font-weight: bold;
            animation: slideIn 0.5s ease;
            box-shadow: 0 5px 15px rgba(0,200,81,0.3);
        }
        @keyframes slideIn {
            from { transform: translateY(-100px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        .status-icon {
            font-size: 1.5em;
            margin-right: 10px;
        }
        .refresh-indicator {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: white;
            padding: 15px 20px;
            border-radius: 50px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            font-weight: 500;
        }
        .loading {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🖥️ GPU Cluster Monitor</h1>
        <div class="timestamp" id="timestamp">Loading...</div>
        
        <div id="alert-container"></div>
        
        <div class="server-grid" id="server-grid">
            <div class="server-card">
                <div class="server-name">Loading...</div>
            </div>
        </div>
        
        <div class="refresh-indicator">
            <span id="refresh-icon">🔄</span>
            <span id="refresh-text">Auto-refresh: 10s</span>
        </div>
    </div>
    
    <script>
        function updateDisplay(data) {
            // Update timestamp
            document.getElementById('timestamp').textContent = 
                'Last updated: ' + data.timestamp;
            
            // Check for free GPUs
            let freeGPUs = [];
            let html = '';
            
            for (let server in data.servers) {
                let serverData = data.servers[server];
                let cardClass = serverData.status === 'offline' ? 'offline' : '';
                
                html += `<div class="server-card ${cardClass}">`;
                html += `<div class="server-name">${server.toUpperCase()}</div>`;
                
                if (serverData.status === 'offline') {
                    html += '<div>❌ Server Offline</div>';
                } else if (serverData.gpus) {
                    serverData.gpus.forEach(gpu => {
                        let gpuClass = '';
                        let icon = '';
                        
                        if (gpu.status === 'FREE') {
                            gpuClass = 'gpu-free';
                            icon = '✅';
                            freeGPUs.push(server + ' ' + gpu.id);
                        } else if (gpu.status === 'PARTIAL') {
                            gpuClass = 'gpu-partial';
                            icon = '⚡';
                        } else {
                            gpuClass = 'gpu-busy';
                            icon = '🔥';
                        }
                        
                        html += `<div class="gpu-item ${gpuClass}">`;
                        html += `<div><span class="status-icon">${icon}</span>${gpu.id}</div>`;
                        html += `<div class="gpu-stats">`;
                        html += `<span class="stat">Mem: ${gpu.mem_percent}%</span>`;
                        html += `<span class="stat">Util: ${gpu.gpu_util}%</span>`;
                        html += `<span class="stat">Temp: ${gpu.temp}°C</span>`;
                        html += `</div>`;
                        
                        if (gpu.users && gpu.users !== 'none') {
                            html += `<div class="user-info">👤 ${gpu.users}</div>`;
                        } else if (gpu.status === 'FREE') {
                            html += `<div class="user-info">🎉 AVAILABLE</div>`;
                        }
                        html += `</div>`;
                    });
                }
                html += `</div>`;
            }
            
            document.getElementById('server-grid').innerHTML = html;
            
            // Update alert banner
            let alertHtml = '';
            if (freeGPUs.length > 0) {
                alertHtml = `<div class="alert-banner">
                    🎉 FREE GPU(S) AVAILABLE: ${freeGPUs.join(', ')} 🎉
                </div>`;
                
                // Update page title to show alert
                document.title = `(${freeGPUs.length} FREE) GPU Monitor`;
                
                // Optional: Play notification sound
                // new Audio('notification.mp3').play();
            } else {
                document.title = 'GPU Cluster Monitor';
            }
            document.getElementById('alert-container').innerHTML = alertHtml;
        }
        
        function fetchStatus() {
            document.getElementById('refresh-icon').classList.add('loading');
            
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    updateDisplay(data);
                    document.getElementById('refresh-icon').classList.remove('loading');
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                    document.getElementById('refresh-icon').classList.remove('loading');
                });
        }
        
        // Initial fetch
        fetchStatus();
        
        // Auto-refresh every 10 seconds
        setInterval(fetchStatus, 10000);
    </script>
</body>
</html>
'''

def check_gpu_server(server):
    """Check GPU status on a remote server"""
    try:
        cmd = f"""ssh -o ConnectTimeout={SSH_TIMEOUT} -o StrictHostKeyChecking=no {server} 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null'"""
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=SSH_TIMEOUT+2)
        
        if result.returncode != 0:
            return {"status": "offline"}
        
        lines = result.stdout.strip().split('\n')
        gpus = []
        
        for line in lines:
            if line:
                parts = line.split(',')
                if len(parts) >= 5:
                    index = parts[0].strip()
                    mem_used = int(parts[1].strip())
                    mem_total = int(parts[2].strip())
                    gpu_util = int(parts[3].strip())
                    temp = int(parts[4].strip())
                    
                    mem_percent = int(mem_used * 100 / mem_total) if mem_total > 0 else 0
                    
                    # Get user info
                    user_cmd = f"""ssh -o ConnectTimeout={SSH_TIMEOUT} -o StrictHostKeyChecking=no {server} "nvidia-smi | grep -A 20 'Processes:' | grep '^ *[0-9]' | awk '{{print \\$3}}' | xargs -I {{}} ps -o user= -p {{}} 2>/dev/null | head -1" """
                    user_result = subprocess.run(user_cmd, shell=True, capture_output=True, text=True, timeout=SSH_TIMEOUT+2)
                    users = user_result.stdout.strip() if user_result.returncode == 0 else "none"
                    
                    # Determine status
                    if mem_percent < 5:
                        status = "FREE"
                    elif mem_percent < 50:
                        status = "PARTIAL"
                    else:
                        status = "BUSY"
                    
                    gpus.append({
                        "id": f"GPU{index}",
                        "status": status,
                        "mem_percent": mem_percent,
                        "gpu_util": gpu_util,
                        "temp": temp,
                        "users": users if users else "none"
                    })
        
        return {"status": "online", "gpus": gpus}
    
    except Exception as e:
        print(f"Error checking {server}: {e}")
        return {"status": "offline"}

def update_gpu_status():
    """Background thread to update GPU status"""
    global gpu_status, last_update
    
    while True:
        new_status = {}
        for server in SERVERS:
            new_status[server] = check_gpu_server(server)
        
        with lock:
            gpu_status = new_status
            last_update = datetime.now()
        
        time.sleep(UPDATE_INTERVAL)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def api_status():
    with lock:
        return jsonify({
            "servers": gpu_status,
            "timestamp": last_update.strftime("%Y-%m-%d %H:%M:%S") if last_update else "Never"
        })

def main():
    parser = argparse.ArgumentParser(description='GPU Cluster Web Monitor')
    parser.add_argument('--port', type=int, default=5000, help='Port to run on (default: 5000)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    args = parser.parse_args()
    
    # Start background updater
    updater = threading.Thread(target=update_gpu_status, daemon=True)
    updater.start()
    
    print(f"🚀 GPU Cluster Web Monitor starting on http://{args.host}:{args.port}")
    print(f"📡 Monitoring servers: {', '.join(SERVERS)}")
    print(f"🔄 Update interval: {UPDATE_INTERVAL} seconds")
    print("\nPress Ctrl+C to stop")
    
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == '__main__':
    main()
