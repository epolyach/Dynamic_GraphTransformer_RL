#!/bin/bash

echo "Testing GPU server access..."
echo "================================"

servers=("gpu1.sedan.pro" "gpu2.sedan.pro" "gpu3.sedan.pro")

for server in "${servers[@]}"; do
    echo -n "Testing $server: "
    if timeout 3 ssh -o ConnectTimeout=2 -o StrictHostKeyChecking=no "$server" "echo 'OK'" 2>/dev/null | grep -q "OK"; then
        echo "✅ Connected"
    else
        echo "❌ Cannot connect (check SSH keys or network)"
    fi
done

echo ""
echo "If connections fail, you need to:"
echo "1. Generate SSH key: ssh-keygen -t rsa"
echo "2. Copy to servers: ssh-copy-id username@gpu1.sedan.pro"
echo "3. Repeat for all servers"
