#!/bin/bash

# Script để mount EBS volume
# Sử dụng: ./scripts/mount-ebs.sh

set -e

echo "💾 Mounting EBS volume..."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Detect EBS device
DEVICE=""
if [ -b /dev/nvme1n1 ]; then
    DEVICE="/dev/nvme1n1"
elif [ -b /dev/xvdf ]; then
    DEVICE="/dev/xvdf"
elif [ -b /dev/sdf ]; then
    DEVICE="/dev/sdf"
else
    echo -e "${RED}❌ No EBS volume detected!${NC}"
    echo "Available devices:"
    lsblk
    exit 1
fi

MOUNT_POINT="/mnt/ebs-data"

echo -e "${YELLOW}📦 Detected device: $DEVICE${NC}"

# Check if already mounted
if mountpoint -q "$MOUNT_POINT"; then
    echo -e "${GREEN}✓ Volume already mounted at $MOUNT_POINT${NC}"
    df -h | grep $MOUNT_POINT
    exit 0
fi

# Check if device has filesystem
if ! sudo file -s $DEVICE | grep -q "filesystem"; then
    echo -e "${YELLOW}⚠ Device $DEVICE doesn't have a filesystem. Formatting...${NC}"
    read -p "This will erase all data on $DEVICE. Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    sudo mkfs -t ext4 $DEVICE
    echo -e "${GREEN}✓ Device formatted${NC}"
fi

# Create mount point
sudo mkdir -p $MOUNT_POINT

# Mount volume
echo -e "${YELLOW}📌 Mounting $DEVICE to $MOUNT_POINT...${NC}"
sudo mount $DEVICE $MOUNT_POINT

# Set permissions
sudo chown -R $USER:$USER $MOUNT_POINT

# Add to fstab for auto-mount on reboot
if ! grep -q "$DEVICE.*$MOUNT_POINT" /etc/fstab; then
    echo -e "${YELLOW}📝 Adding to /etc/fstab for auto-mount...${NC}"
    UUID=$(sudo blkid -s UUID -o value $DEVICE)
    echo "UUID=$UUID $MOUNT_POINT ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
    echo -e "${GREEN}✓ Added to fstab${NC}"
fi

# Verify
if mountpoint -q "$MOUNT_POINT"; then
    echo -e "${GREEN}✅ Volume mounted successfully!${NC}"
    echo ""
    echo "Mount info:"
    df -h | grep $MOUNT_POINT
    echo ""
    echo "You can now use: $MOUNT_POINT"
else
    echo -e "${RED}❌ Mount failed!${NC}"
    exit 1
fi

