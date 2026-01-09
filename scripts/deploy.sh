#!/bin/bash

# Script deploy tự động cho EC2
# Sử dụng: ./scripts/deploy.sh

set -e  # Exit on error

echo "🚀 Starting deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/home/ubuntu/law-chatbot/law-chatbot"
COMPOSE_FILE="docker-compose.yml"

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ docker-compose not found. Installing...${NC}"
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Navigate to project directory
cd "$PROJECT_DIR" || {
    echo -e "${RED}❌ Project directory not found: $PROJECT_DIR${NC}"
    exit 1
}

echo -e "${GREEN}✓ Found project directory${NC}"

# Create necessary directories
mkdir -p ../law-chatbot-backend/storage/chroma
mkdir -p ../law-chatbot-backend/storage/logs
mkdir -p ../law-chatbot-backend/storage/data

# Set permissions
sudo chown -R $USER:$USER ../law-chatbot-backend/storage

# Stop existing containers
echo -e "${YELLOW}🛑 Stopping existing containers...${NC}"
docker-compose -f "$COMPOSE_FILE" down || true

# Pull latest images (if using pre-built images)
# docker-compose -f "$COMPOSE_FILE" pull

# Build and start containers
echo -e "${YELLOW}🔨 Building and starting containers...${NC}"
docker-compose -f "$COMPOSE_FILE" up -d --build

# Wait for services to be ready
echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 15

# Health check
echo -e "${YELLOW}🏥 Running health checks...${NC}"

# Check backend
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Backend is healthy${NC}"
else
    echo -e "${RED}❌ Backend health check failed${NC}"
    docker-compose -f "$COMPOSE_FILE" logs backend
    exit 1
fi

# Check ChromaDB
if curl -f http://localhost:8001/api/v1/heartbeat > /dev/null 2>&1; then
    echo -e "${GREEN}✓ ChromaDB is healthy${NC}"
else
    echo -e "${YELLOW}⚠ ChromaDB health check failed (may need more time)${NC}"
fi

# Clean up old images
echo -e "${YELLOW}🧹 Cleaning up old Docker images...${NC}"
docker image prune -f

# Show status
echo -e "${GREEN}📊 Container status:${NC}"
docker-compose -f "$COMPOSE_FILE" ps

echo -e "${GREEN}✅ Deployment completed successfully!${NC}"

