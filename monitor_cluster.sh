#!/bin/bash
echo "=== 🪶 Feather Molt Cluster Monitor ==="
echo "Initializing ASITOP across unified memory..."

# Install asitop locally if missing
if ! command -v asitop &> /dev/null; then
    pip install asitop >/dev/null 2>&1
fi

echo "Worker Nodes Memory Pressure:"
for ip in 10.0.0.63 10.0.0.19 10.0.0.118; do
   echo -n "Node $ip: "
   ssh -i ~/.ssh/ubuntu-mac-cluster_user-admin -o StrictHostKeyChecking=no cluster_user@$ip "vm_stat | grep 'Pages active'" 2>/dev/null || echo "Offline"
done
echo "----------------------------------------"
echo "Launching interactive ASITOP Dashboard on Head Node (Press Q to quit)..."
sudo asitop
