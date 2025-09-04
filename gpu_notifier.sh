#!/bin/bash

# GPU Availability Notifier
# Monitors GPU cluster and sends notifications when GPUs become free

SERVERS=("gpu1.sedan.pro" "gpu2.sedan.pro" "gpu3.sedan.pro")
CHECK_INTERVAL=30  # seconds
SSH_TIMEOUT=5

# State tracking
PREV_STATE_FILE="/tmp/.gpu_notifier_state"

# Notification methods (configure as needed)
SEND_EMAIL=false
EMAIL_TO="your-email@example.com"

SEND_SLACK=false
SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

SEND_TERMINAL_BELL=true
SEND_DESKTOP_NOTIFY=true

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to check if GPU is free on server
check_server() {
    local server=$1
    ssh -o ConnectTimeout=${SSH_TIMEOUT} -o StrictHostKeyChecking=no "$server" \
        'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | \
         awk -F, "{if (\$1/\$2 < 0.05) print \"FREE\"; else print \"BUSY\"}"' 2>/dev/null || echo "OFFLINE"
}

# Send notification
send_notification() {
    local message=$1
    local title="GPU Available!"
    
    echo -e "${GREEN}🔔 $message${NC}"
    
    # Terminal bell
    if [ "$SEND_TERMINAL_BELL" = true ]; then
        echo -e "\a"
    fi
    
    # Desktop notification (Linux)
    if [ "$SEND_DESKTOP_NOTIFY" = true ] && command -v notify-send &>/dev/null; then
        notify-send -u critical "$title" "$message" -i computer
    fi
    
    # macOS notification
    if [ "$SEND_DESKTOP_NOTIFY" = true ] && command -v osascript &>/dev/null; then
        osascript -e "display notification \"$message\" with title \"$title\" sound name \"Glass\""
    fi
    
    # Email notification
    if [ "$SEND_EMAIL" = true ] && command -v mail &>/dev/null; then
        echo "$message" | mail -s "$title" "$EMAIL_TO"
    fi
    
    # Slack notification
    if [ "$SEND_SLACK" = true ] && [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\":computer: $message\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null
    fi
}

# Load previous state
load_prev_state() {
    if [ -f "$PREV_STATE_FILE" ]; then
        source "$PREV_STATE_FILE"
    else
        declare -gA PREV_STATE
    fi
}

# Save current state
save_state() {
    > "$PREV_STATE_FILE"
    for server in "${!CURRENT_STATE[@]}"; do
        echo "PREV_STATE[$server]=${CURRENT_STATE[$server]}" >> "$PREV_STATE_FILE"
    done
}

# Main monitoring loop
monitor() {
    declare -gA PREV_STATE
    declare -gA CURRENT_STATE
    
    load_prev_state
    
    echo -e "${GREEN}🖥️  GPU Notifier Started${NC}"
    echo "Monitoring: ${SERVERS[*]}"
    echo "Check interval: ${CHECK_INTERVAL}s"
    echo "Notifications: Bell=$SEND_TERMINAL_BELL Desktop=$SEND_DESKTOP_NOTIFY"
    echo ""
    
    while true; do
        local free_servers=()
        local newly_free=()
        
        # Check each server
        for server in "${SERVERS[@]}"; do
            status=$(check_server "$server")
            CURRENT_STATE[$server]=$status
            
            if [ "$status" = "FREE" ]; then
                free_servers+=("$server")
                
                # Check if newly free
                if [ "${PREV_STATE[$server]}" != "FREE" ]; then
                    newly_free+=("$server")
                fi
            fi
        done
        
        # Display current status
        printf "\r[%s] " "$(date '+%H:%M:%S')"
        for server in "${SERVERS[@]}"; do
            server_short=$(echo "$server" | cut -d'.' -f1)
            status=${CURRENT_STATE[$server]}
            
            case "$status" in
                "FREE")
                    printf "${GREEN}%s:✅${NC} " "$server_short"
                    ;;
                "BUSY")
                    printf "${RED}%s:🔥${NC} " "$server_short"
                    ;;
                "OFFLINE")
                    printf "${YELLOW}%s:❌${NC} " "$server_short"
                    ;;
            esac
        done
        
        # Send notifications for newly free GPUs
        if [ ${#newly_free[@]} -gt 0 ]; then
            echo ""  # New line before notification
            message="GPU(s) now available: ${newly_free[*]}"
            send_notification "$message"
        fi
        
        # Save state
        save_state
        
        # Wait for next check
        sleep "$CHECK_INTERVAL"
    done
}

# Cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Stopping GPU notifier...${NC}"
    rm -f "$PREV_STATE_FILE"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Parse arguments
case "${1:-}" in
    "config")
        echo "Configuration file: gpu_notifier.conf"
        cat > gpu_notifier.conf << 'CONFIG'
# GPU Notifier Configuration
# Edit this file to customize notifications

# Email settings
SEND_EMAIL=false
EMAIL_TO="your-email@example.com"

# Slack settings
SEND_SLACK=false
SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# Local notifications
SEND_TERMINAL_BELL=true
SEND_DESKTOP_NOTIFY=true

# Check interval (seconds)
CHECK_INTERVAL=30

# Servers to monitor
SERVERS=("gpu1.sedan.pro" "gpu2.sedan.pro" "gpu3.sedan.pro")
CONFIG
        echo "Configuration file created. Edit it and source it before running."
        ;;
    "test")
        echo "Testing notification..."
        send_notification "Test notification - GPU monitoring is working!"
        ;;
    *)
        # Load config if exists
        [ -f gpu_notifier.conf ] && source gpu_notifier.conf
        monitor
        ;;
esac
