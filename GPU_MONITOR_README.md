# 🖥️ GPU Cluster Monitoring Suite

A comprehensive set of tools for monitoring GPU usage across multiple servers (gpu1.sedan.pro, gpu2.sedan.pro, gpu3.sedan.pro).

## 📦 Components

### 1. **gpu_cluster_monitor.sh** - Terminal-based Multi-Server Monitor
Comprehensive bash script for monitoring all GPU servers from the command line.

#### Usage:
```bash
# Single check with detailed info
./gpu_cluster_monitor.sh

# Continuous monitoring (updates every 10 seconds)
./gpu_cluster_monitor.sh watch

# Quick one-line status
./gpu_cluster_monitor.sh quick

# Help
./gpu_cluster_monitor.sh help
```

#### Features:
- ✅ Color-coded status (FREE/BUSY/OFFLINE)
- 👥 Shows current users
- 📊 Displays GPU metrics (memory, utilization, temperature)
- 🔔 Visual alerts when GPUs are free
- 🔄 Auto-refresh mode

### 2. **gpu_web_monitor.py** - Web Dashboard
Flask-based web interface for remote monitoring.

#### Setup:
```bash
# Install Flask if not already installed
pip install flask

# Start the web server
python3 gpu_web_monitor.py --port 5000

# Access from browser
# http://your-server-ip:5000
```

#### Features:
- 🌐 Access from any browser
- 📱 Mobile-responsive design
- 🔄 Auto-refresh every 10 seconds
- 🎨 Beautiful visual interface
- 🔔 Browser title alerts when GPUs are free
- 📊 Real-time status updates

### 3. **gpu_notifier.sh** - Availability Notifier
Background service that sends notifications when GPUs become available.

#### Usage:
```bash
# Create configuration file
./gpu_notifier.sh config

# Edit configuration (optional)
nano gpu_notifier.conf

# Test notifications
./gpu_notifier.sh test

# Start monitoring
./gpu_notifier.sh

# Run in background
nohup ./gpu_notifier.sh > gpu_notifier.log 2>&1 &
```

#### Notification Methods:
- 🔔 Terminal bell
- 🖥️ Desktop notifications (Linux/macOS)
- 📧 Email alerts (configure SMTP)
- 💬 Slack integration (webhook)

## 🚀 Quick Start

### Basic Monitoring
```bash
# Check current status
./gpu_cluster_monitor.sh quick

# Watch for changes
./gpu_cluster_monitor.sh watch
```

### Web Monitoring
```bash
# Start web server
python3 gpu_web_monitor.py &

# Open in browser
xdg-open http://localhost:5000  # Linux
open http://localhost:5000       # macOS
```

### Background Notifications
```bash
# Start notifier in background
nohup ./gpu_notifier.sh > /dev/null 2>&1 &

# Check if running
ps aux | grep gpu_notifier
```

## 🔧 Configuration

### SSH Setup
Ensure passwordless SSH access to all GPU servers:
```bash
# Generate SSH key if needed
ssh-keygen -t rsa

# Copy to GPU servers
ssh-copy-id gpu1.sedan.pro
ssh-copy-id gpu2.sedan.pro
ssh-copy-id gpu3.sedan.pro
```

### Customize Servers
Edit the SERVERS array in any script:
```bash
SERVERS=("gpu1.sedan.pro" "gpu2.sedan.pro" "gpu3.sedan.pro")
```

## 📊 Status Indicators

| Icon | Status | Description |
|------|--------|-------------|
| ✅ | FREE | GPU memory < 5% used |
| ⚡ | PARTIAL | GPU memory 5-50% used |
| 🔥 | BUSY | GPU memory > 50% used |
| ❌ | OFFLINE | Server unreachable |

## 🎯 Use Cases

### 1. Quick Check Before Training
```bash
./gpu_cluster_monitor.sh quick
# Output: [gpu1:🔥][gpu2:✅][gpu3:⚡] 1 FREE
```

### 2. Wait for Available GPU
```bash
# Terminal 1: Start notifier
./gpu_notifier.sh

# Terminal 2: Do other work
# You'll get a notification when GPU is free
```

### 3. Remote Monitoring
```bash
# On server
python3 gpu_web_monitor.py --host 0.0.0.0 --port 8080

# From laptop/phone
# Browse to: http://server-ip:8080
```

### 4. Team Dashboard
Set up the web monitor on a shared server for team visibility:
```bash
# Create systemd service (Linux)
sudo tee /etc/systemd/system/gpu-monitor.service << END
[Unit]
Description=GPU Cluster Web Monitor
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PWD
ExecStart=/usr/bin/python3 $PWD/gpu_web_monitor.py --port 5000
Restart=always

[Install]
WantedBy=multi-user.target
END

sudo systemctl enable gpu-monitor
sudo systemctl start gpu-monitor
```

## 🛠️ Troubleshooting

### SSH Connection Issues
```bash
# Test SSH connection
ssh -v gpu1.sedan.pro echo "Connected"

# Check SSH config
cat ~/.ssh/config
```

### Permission Denied
```bash
# Make scripts executable
chmod +x gpu_*.sh
chmod +x gpu_*.py
```

### Port Already in Use (Web Monitor)
```bash
# Use different port
python3 gpu_web_monitor.py --port 8080
```

## 📝 Tips

1. **Aliases**: Add to ~/.bashrc for quick access:
   ```bash
   alias gpu='~/path/to/gpu_cluster_monitor.sh quick'
   alias gpuw='~/path/to/gpu_cluster_monitor.sh watch'
   ```

2. **Cron Job**: Regular status emails:
   ```bash
   # Add to crontab
   0 */2 * * * /path/to/gpu_cluster_monitor.sh | mail -s "GPU Status" you@email.com
   ```

3. **tmux/screen**: Keep monitoring running:
   ```bash
   tmux new -s gpu-monitor
   ./gpu_cluster_monitor.sh watch
   # Ctrl+B, D to detach
   # tmux attach -t gpu-monitor to reattach
   ```

## 📄 License
These scripts are provided as-is for GPU cluster monitoring.

## 🤝 Contributing
Feel free to modify and extend these scripts for your needs!
