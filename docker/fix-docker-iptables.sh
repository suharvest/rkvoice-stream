#!/bin/bash
# Fix Docker iptables issue on RK3576.
#
# The RK3576 kernel (6.1.99) lacks the iptable_raw module.
# Docker 29.x requires it for "direct access filtering".
#
# Solution: disable Docker's iptables and use --network=host for containers.
#
# Run this ONCE on the RK3576 device (as root):
#   sudo bash fix-docker-iptables.sh

set -euo pipefail

# Ensure iptables-legacy is the default (not nft)
update-alternatives --set iptables /usr/sbin/iptables-legacy 2>/dev/null || true
update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy 2>/dev/null || true

# Configure Docker to not manage iptables (avoids iptable_raw requirement)
mkdir -p /etc/docker
cat > /etc/docker/daemon.json << 'EOF'
{
  "iptables": false,
  "registry-mirrors": ["https://docker.1ms.run", "https://docker.xuanyuan.me"]
}
EOF

# Restart Docker
systemctl restart docker
sleep 3

# Verify Docker is running
if docker ps >/dev/null 2>&1; then
    echo "OK: Docker is running"
    echo "NOTE: Use --network=host for all containers (port mapping won't work)"
else
    echo "ERROR: Docker failed to start. Check: journalctl -xeu docker.service"
    exit 1
fi
