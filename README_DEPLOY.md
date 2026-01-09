# Quick Start - Deploy to EC2

Hướng dẫn nhanh để deploy lên EC2.

## 🚀 Quick Deploy

### 1. Tạo EC2 Instance

- AMI: Ubuntu 22.04 LTS
- Type: t3.micro (hoặc t2.micro cho free tier)
- Security Group: Mở ports 22, 80, 443, 8000, 3000, 8001

### 2. Tạo EBS Volume

- Size: 20GB
- Type: gp3
- **Quan trọng**: Phải cùng Availability Zone với EC2!

### 3. Setup trên EC2

```bash
# SSH vào EC2
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# Format và mount EBS
sudo mkfs -t ext4 /dev/nvme1n1
sudo mkdir -p /mnt/ebs-data
sudo mount /dev/nvme1n1 /mnt/ebs-data
echo '/dev/nvme1n1 /mnt/ebs-data ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab

# Cài Docker
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone code
cd /mnt/ebs-data
git clone YOUR_REPO_URL law-chatbot
cd law-chatbot/law-chatbot

# Tạo .env file
nano .env  # Thêm các biến môi trường cần thiết

# Deploy
chmod +x ../scripts/deploy.sh
../scripts/deploy.sh
```

### 4. Setup CI/CD (Optional)

1. Tạo SSH key trên EC2: `ssh-keygen -t ed25519 -f ~/.ssh/github_actions_key`
2. Copy private key: `cat ~/.ssh/github_actions_key`
3. Thêm vào GitHub Secrets:
   - `EC2_INSTANCE_IP`: IP của EC2
   - `EC2_SSH_KEY`: Private key
   - `EC2_USER`: ubuntu
4. Push code lên main branch → Tự động deploy!

## 📖 Chi tiết

Xem file [DEPLOY_GUIDE.md](./DEPLOY_GUIDE.md) để biết hướng dẫn đầy đủ.

