#!/bin/bash

# Script setup ban đầu cho EC2 instance
# Chạy script này sau khi SSH vào EC2 lần đầu

set -e

echo "🔧 Setting up EC2 instance for Law Chatbot..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Update system
echo -e "${YELLOW}📦 Updating system packages...${NC}"
sudo apt update && sudo apt upgrade -y

# Install essential tools
echo -e "${YELLOW}📦 Installing essential tools...${NC}"
sudo apt install -y \
    curl \
    wget \
    git \
    unzip \
    htop \
    nano \
    ufw

# Install Docker
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}🐳 Installing Docker...${NC}"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo -e "${GREEN}✓ Docker installed${NC}"
else
    echo -e "${GREEN}✓ Docker already installed${NC}"
fi

# Install Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}🐳 Installing Docker Compose...${NC}"
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo -e "${GREEN}✓ Docker Compose installed${NC}"
else
    echo -e "${GREEN}✓ Docker Compose already installed${NC}"
fi

# Setup firewall
echo -e "${YELLOW}🔥 Configuring firewall...${NC}"
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw allow 8000/tcp # Backend API
sudo ufw allow 3000/tcp # Frontend
sudo ufw allow 8001/tcp # ChromaDB
sudo ufw --force enable
echo -e "${GREEN}✓ Firewall configured${NC}"

# Create directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p ~/law-chatbot
mkdir -p /mnt/ebs-data/law-chatbot
sudo chown -R $USER:$USER /mnt/ebs-data/law-chatbot
echo -e "${GREEN}✓ Directories created${NC}"

# Setup swap (nếu cần cho t2.micro)
if [ ! -f /swapfile ]; then
    echo -e "${YELLOW}💾 Creating swap file (2GB)...${NC}"
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo -e "${GREEN}✓ Swap file created${NC}"
fi

echo -e "${GREEN}✅ Setup completed!${NC}"
echo ""
echo "Next steps:"
echo "1. Mount EBS volume (if not already mounted)"
echo "2. Clone your repository"
echo "3. Create .env file"
echo "4. Run deploy script"

