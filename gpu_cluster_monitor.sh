#!/bin/bash

# Multi-Server GPU Cluster Monitor
# Monitors gpu1.sedan.pro, gpu2.sedan.pro, gpu3.sedan.pro
# Shows load, users, and alerts when GPUs are free

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Servers to monitor
SERVERS=("gpu1.sedan.pro" "gpu2.sedan.pro" "gpu3.sedan.pro")

# SSH timeout in seconds
SSH_TIMEOUT=5

# Function to check GPU on a server
check_gpu_server() {
    local server=$1
    local ssh_cmd="ssh -o ConnectTimeout=${SSH_TIMEOUT} -o StrictHostKeyChecking=no"
    
    # Test connection first
    if ! $ssh_cmd "$server" "echo 'connected'" &>/dev/null; then
        echo "OFFLINE"
        return 1
    fi
    
    # Get GPU info
    $ssh_cmd "$server" 'bash -s' << 'REMOTE_SCRIPT' 2>/dev/null
    if ! command -v nvidia-smi &>/dev/null; then
        echo "NO_GPU"
        exit 1
    fi
    
    # Get GPU data
    gpu_data=$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
    if [ -z "$gpu_data" ]; then
        echo "GPU_ERROR"
        exit 1
    fi
    
    # Process each GPU
    while IFS=',' read -r index name mem_used mem_total gpu_util temp; do
        # Clean up values
        index=$(echo "$index" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)
        gpu_util=$(echo "$gpu_util" | xargs)
        temp=$(echo "$temp" | xargs)
        
        # Calculate memory percentage
        if [ "$mem_total" -gt 0 ] 2>/dev/null; then
            mem_percent=$((mem_used * 100 / mem_total))
        else
            mem_percent=0
        fi
        
        # Get processes using this GPU
        processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep -v "^$")
        
        users=""
        if [ -n "$processes" ]; then
            # Get unique users
            users=$(echo "$processes" | while IFS=',' read -r gpu_uuid pid; do
                pid=$(echo "$pid" | xargs)
                if [ -n "$pid" ] && ps -p "$pid" &>/dev/null; then
                    ps -o user= -p "$pid" 2>/dev/null
                fi
            done | sort -u | tr '\n' ',' | sed 's/,$//')
        fi
        
        # Determine status
        if [ -z "$users" ] || [ "$mem_percent" -lt 5 ]; then
            status="FREE"
        elif [ "$mem_percent" -lt 50 ]; then
            status="PARTIAL"
        else
            status="BUSY"
        fi
        
        # Output format: GPU_INDEX|STATUS|MEM_PERCENT|GPU_UTIL|TEMP|USERS
        echo "GPU${index}|${status}|${mem_percent}|${gpu_util}|${temp}|${users:-none}"
    done <<< "$gpu_data"
REMOTE_SCRIPT
}

# Function to display server status
display_server_status() {
    local server=$1
    local result=$2
    local server_name=$(echo "$server" | cut -d'.' -f1)
    
    echo -e "\n${BOLD}${BLUE}━━━ ${server_name^^} ━━━${NC}"
    
    if [ "$result" = "OFFLINE" ]; then
        echo -e "${RED}❌ Server Offline${NC}"
        return 1
    elif [ "$result" = "NO_GPU" ]; then
        echo -e "${YELLOW}⚠️  No GPU available${NC}"
        return 1
    elif [ "$result" = "GPU_ERROR" ]; then
        echo -e "${YELLOW}⚠️  GPU Error${NC}"
        return 1
    fi
    
    # Parse and display GPU info
    local has_free_gpu=false
    while IFS='|' read -r gpu_id status mem_percent gpu_util temp users; do
        # Set color based on status
        case "$status" in
            "FREE")
                color="${GREEN}"
                icon="✅"
                has_free_gpu=true
                ;;
            "PARTIAL")
                color="${YELLOW}"
                icon="⚡"
                ;;
            "BUSY")
                color="${RED}"
                icon="🔥"
                ;;
        esac
        
        # Format output
        printf "${color}${icon} %s: %3d%% mem, %3d%% util, %2d°C${NC}" \
               "$gpu_id" "$mem_percent" "$gpu_util" "$temp"
        
        if [ "$users" != "none" ]; then
            echo -e " ${CYAN}[${users}]${NC}"
        else
            echo -e " ${GREEN}[AVAILABLE]${NC}"
        fi
    done <<< "$result"
    
    return $([ "$has_free_gpu" = true ] && echo 0 || echo 1)
}

# Function for continuous monitoring
monitor_continuous() {
    local interval=${1:-10}  # Default 10 seconds
    
    while true; do
        clear
        monitor_once
        echo -e "\n${CYAN}Refreshing every ${interval} seconds... (Ctrl+C to stop)${NC}"
        sleep "$interval"
    done
}

# Function for single check
monitor_once() {
    echo -e "${BOLD}${CYAN}╔══════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║    GPU CLUSTER MONITOR               ║${NC}"
    echo -e "${BOLD}${CYAN}║    $(date '+%Y-%m-%d %H:%M:%S')         ║${NC}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════╝${NC}"
    
    local free_servers=()
    local busy_servers=()
    local offline_servers=()
    
    for server in "${SERVERS[@]}"; do
        result=$(check_gpu_server "$server")
        display_server_status "$server" "$result"
        
        # Categorize servers
        if [ "$result" = "OFFLINE" ]; then
            offline_servers+=("$server")
        elif echo "$result" | grep -q "FREE"; then
            free_servers+=("$server")
        else
            busy_servers+=("$server")
        fi
    done
    
    # Summary
    echo -e "\n${BOLD}${CYAN}═══ SUMMARY ═══${NC}"
    
    if [ ${#free_servers[@]} -gt 0 ]; then
        echo -e "${GREEN}🎉 FREE GPUS AVAILABLE on:${NC}"
        for server in "${free_servers[@]}"; do
            echo -e "   ${GREEN}✓ $(echo $server | cut -d'.' -f1)${NC}"
        done
    fi
    
    if [ ${#busy_servers[@]} -gt 0 ]; then
        echo -e "${YELLOW}⚡ Busy servers: ${busy_servers[*]}${NC}"
    fi
    
    if [ ${#offline_servers[@]} -gt 0 ]; then
        echo -e "${RED}❌ Offline: ${offline_servers[*]}${NC}"
    fi
    
    # Alert if GPUs are free
    if [ ${#free_servers[@]} -gt 0 ]; then
        echo -e "\n${GREEN}${BOLD}🔔 ALERT: FREE GPU(S) AVAILABLE! 🔔${NC}"
        # Optional: Play sound alert (if terminal bell is enabled)
        echo -e "\a"
    fi
}

# Function to get quick status in one line
quick_status() {
    local free_count=0
    local output=""
    
    for server in "${SERVERS[@]}"; do
        server_short=$(echo "$server" | cut -d'.' -f1)
        result=$(check_gpu_server "$server" 2>/dev/null)
        
        if [ "$result" = "OFFLINE" ]; then
            output="${output}[${server_short}:❌]"
        elif echo "$result" | grep -q "FREE"; then
            output="${output}[${server_short}:✅]"
            ((free_count++))
        else
            output="${output}[${server_short}:🔥]"
        fi
    done
    
    echo -n "$output"
    if [ $free_count -gt 0 ]; then
        echo -e " ${GREEN}${free_count} FREE${NC}"
    else
        echo -e " ${RED}ALL BUSY${NC}"
    fi
}

# Parse command line arguments
case "${1:-}" in
    "watch"|"-w")
        interval=${2:-10}
        monitor_continuous "$interval"
        ;;
    "quick"|"-q")
        quick_status
        ;;
    "help"|"-h"|"--help")
        echo "GPU Cluster Monitor - Monitor multiple GPU servers"
        echo ""
        echo "Usage:"
        echo "  $0              # Single check with detailed info"
        echo "  $0 watch [sec]  # Continuous monitoring (default 10s)"
        echo "  $0 quick        # Quick one-line status"
        echo "  $0 help         # Show this help"
        echo ""
        echo "Servers monitored: ${SERVERS[*]}"
        ;;
    *)
        monitor_once
        ;;
esac
