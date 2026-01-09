# Hướng dẫn Deploy lên EC2 với EBS

Hướng dẫn chi tiết để deploy ứng dụng Law Chatbot lên AWS EC2 với EBS volume giá rẻ và thiết lập CI/CD tự động.

## 📋 Mục lục

1. [Chuẩn bị](#chuẩn-bị)
2. [Tạo EC2 Instance](#tạo-ec2-instance)
3. [Tạo và Attach EBS Volume](#tạo-và-attach-ebs-volume)
4. [Cấu hình EC2 Instance](#cấu-hình-ec2-instance)
5. [Thiết lập CI/CD với GitHub Actions](#thiết-lập-cicd-với-github-actions)
6. [Deploy thủ công (nếu cần)](#deploy-thủ-công-nếu-cần)
7. [Troubleshooting](#troubleshooting)

---

## 🛠 Chuẩn bị

### Yêu cầu

- AWS Account
- GitHub repository chứa code
- SSH key pair (hoặc tạo mới trên AWS)
- Kiến thức cơ bản về Docker và Linux

### Chi phí ước tính (tháng)

- **EC2 t2.micro**: ~$8-10/tháng (Free tier: 750 giờ/tháng trong 12 tháng đầu)
- **EBS gp3 20GB**: ~$1.6/tháng
- **Data Transfer**: ~$0.09/GB
- **Tổng**: ~$10-15/tháng (hoặc miễn phí nếu dùng Free tier)

---

## 🖥 Tạo EC2 Instance

### Bước 1: Launch EC2 Instance

1. Đăng nhập vào AWS Console
2. Vào **EC2** → **Instances** → **Launch Instance**
3. Cấu hình:

   **Name**: `law-chatbot-server`

   **AMI**: 
   - Ubuntu 22.04 LTS (miễn phí)
   - Hoặc Amazon Linux 2023

   **Instance Type**: 
   - `t2.micro` (Free tier) - 1 vCPU, 1GB RAM
   - `t3.micro` - 2 vCPU, 1GB RAM (~$8/tháng)
   - `t3.small` - 2 vCPU, 2GB RAM (~$15/tháng) - **Khuyến nghị cho production**

   **Key Pair**: 
   - Chọn key pair có sẵn hoặc tạo mới
   - **Lưu file `.pem` an toàn!**

   **Network Settings**:
   - VPC: Default hoặc tạo mới
   - Subnet: Public subnet
   - Auto-assign Public IP: Enable
   - Security Group: Tạo mới với rules:
     ```
     Type            Protocol    Port Range    Source
     SSH             TCP         22            My IP
     HTTP            TCP         80            Anywhere (0.0.0.0/0)
     HTTPS           TCP         443           Anywhere (0.0.0.0/0)
     Custom TCP      TCP         8000          Anywhere (0.0.0.0/0)  # Backend API
     Custom TCP      TCP         3000          Anywhere (0.0.0.0/0)  # Frontend (nếu cần)
     Custom TCP      TCP         8001          Anywhere (0.0.0.0/0)  # ChromaDB (nếu expose)
     ```

   **Configure Storage**:
   - Root volume: 8GB gp3 (đủ cho OS và Docker)
   - **Không tạo EBS ở đây**, sẽ tạo riêng sau

4. Click **Launch Instance**

### Bước 2: Lấy Public IP

Sau khi instance chạy, lấy **Public IPv4 address** từ EC2 Console.

---

## 💾 Tạo và Attach EBS Volume

### Bước 1: Tạo EBS Volume

1. Vào **EC2** → **Volumes** → **Create Volume**
2. Cấu hình:
   - **Size**: 20GB (hoặc tùy nhu cầu)
   - **Volume Type**: `gp3` (rẻ nhất, $0.08/GB/tháng)
   - **Availability Zone**: **Phải cùng AZ với EC2 instance!**
   - **Encryption**: Optional (khuyến nghị bật)
3. Click **Create Volume**

### Bước 2: Attach Volume vào EC2

1. Chọn volume vừa tạo → **Actions** → **Attach Volume**
2. Chọn EC2 instance của bạn
3. Device name: `/dev/sdf` (hoặc để mặc định)
4. Click **Attach**

### Bước 3: Format và Mount EBS Volume

SSH vào EC2 instance:

```bash
ssh -i your-key.pem ubuntu@YOUR_EC2_IP
```

Chạy các lệnh sau:

```bash
# Kiểm tra volume đã attach
lsblk

# Format volume (CHỈ CHẠY LẦN ĐẦU - sẽ xóa dữ liệu!)
sudo mkfs -t ext4 /dev/nvme1n1  # hoặc /dev/xvdf tùy instance type

# Tạo mount point
sudo mkdir -p /mnt/ebs-data

# Mount volume
sudo mount /dev/nvme1n1 /mnt/ebs-data

# Kiểm tra
df -h

# Mount tự động khi reboot
echo '/dev/nvme1n1 /mnt/ebs-data ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab
```

### Bước 4: Tạo thư mục cho project

```bash
# Tạo thư mục cho project trên EBS
sudo mkdir -p /mnt/ebs-data/law-chatbot
sudo chown -R ubuntu:ubuntu /mnt/ebs-data/law-chatbot

# Tạo symlink (tùy chọn)
mkdir -p ~/law-chatbot
ln -s /mnt/ebs-data/law-chatbot ~/law-chatbot
```

---

## ⚙️ Cấu hình EC2 Instance

### Bước 1: Cài đặt Docker và Docker Compose

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Cài đặt Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Cài đặt Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Logout và login lại để áp dụng group changes
exit
# SSH lại vào instance
```

### Bước 2: Clone code từ GitHub

```bash
cd /mnt/ebs-data/law-chatbot

# Clone repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git .

# Hoặc nếu repo private, cần setup SSH key hoặc Personal Access Token
```

### Bước 3: Tạo file .env

```bash
cd law-chatbot
nano .env
```

Thêm các biến môi trường cần thiết:

```env
# Database
DB_USERNAME=admin
DB_PASSWORD=your_secure_password
DB_NAME=law_chatbot

# ChromaDB
CHROMA_HOST=chromadb
CHROMA_PORT=8000
CHROMA_PERSIST_DIRECTORY=/mnt/ebs-data/law-chatbot/law-chatbot-backend/storage/chroma

# Embedding
EMBEDDING_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key

# LLM
LLM_TYPE=gemini
GEMINI_API_KEY=your_gemini_api_key
```

### Bước 4: Cập nhật docker-compose.yml

Cập nhật paths trong `docker-compose.yml` để trỏ đến EBS volume:

```yaml
volumes:
  - /mnt/ebs-data/law-chatbot/law-chatbot-backend/storage/chroma:/chroma/chroma
  - /mnt/ebs-data/law-chatbot/law-chatbot-backend/storage/logs:/app/storage/logs
```

### Bước 5: Test deploy thủ công

```bash
cd /mnt/ebs-data/law-chatbot/law-chatbot
chmod +x ../scripts/deploy.sh
../scripts/deploy.sh
```

---

## 🔄 Thiết lập CI/CD với GitHub Actions

### Bước 1: Tạo SSH Key cho GitHub Actions

Trên EC2 instance:

```bash
# Tạo SSH key mới (hoặc dùng key hiện có)
ssh-keygen -t ed25519 -C "github-actions" -f ~/.ssh/github_actions_key -N ""

# Thêm public key vào authorized_keys
cat ~/.ssh/github_actions_key.pub >> ~/.ssh/authorized_keys

# Hiển thị private key (copy toàn bộ output)
cat ~/.ssh/github_actions_key
```

### Bước 2: Thêm Secrets vào GitHub

1. Vào GitHub repository → **Settings** → **Secrets and variables** → **Actions**
2. Thêm các secrets sau:

   - **EC2_INSTANCE_IP**: Public IP của EC2 instance
   - **EC2_SSH_KEY**: Nội dung private key (từ bước trên)
   - **EC2_USER**: `ubuntu` (hoặc `ec2-user` nếu dùng Amazon Linux)

### Bước 3: Cập nhật GitHub Actions Workflow

File `.github/workflows/deploy-ec2.yml` đã được tạo sẵn. Kiểm tra và điều chỉnh:

- **AWS_REGION**: Đổi thành region của bạn
- **EC2_USER**: Đổi nếu dùng Amazon Linux (`ec2-user`)
- **Paths**: Kiểm tra paths trong rsync và deploy script

### Bước 4: Test CI/CD

```bash
# Push code lên main branch
git add .
git commit -m "Setup CI/CD"
git push origin main
```

GitHub Actions sẽ tự động chạy và deploy.

---

## 🚀 Deploy thủ công (nếu cần)

Nếu không dùng CI/CD, có thể deploy thủ công:

```bash
# SSH vào EC2
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# Vào thư mục project
cd /mnt/ebs-data/law-chatbot/law-chatbot

# Pull latest code
git pull origin main

# Chạy deploy script
../scripts/deploy.sh
```

---

## 🔧 Troubleshooting

### Vấn đề: Không thể SSH vào EC2

**Giải pháp**:
- Kiểm tra Security Group có mở port 22
- Kiểm tra key file có đúng quyền: `chmod 400 your-key.pem`
- Kiểm tra Public IP có đúng

### Vấn đề: EBS Volume không mount

**Giải pháp**:
```bash
# Kiểm tra volume
lsblk

# Kiểm tra fstab
cat /etc/fstab

# Mount thủ công
sudo mount /dev/nvme1n1 /mnt/ebs-data
```

### Vấn đề: Docker không chạy

**Giải pháp**:
```bash
# Kiểm tra Docker service
sudo systemctl status docker

# Start Docker
sudo systemctl start docker

# Kiểm tra user có trong docker group
groups
```

### Vấn đề: Containers không start

**Giải pháp**:
```bash
# Xem logs
docker-compose logs

# Xem logs của service cụ thể
docker-compose logs backend
docker-compose logs chromadb

# Restart containers
docker-compose restart
```

### Vấn đề: Out of memory

**Giải pháp**:
- Upgrade instance type (t3.small trở lên)
- Hoặc tạo swap file:
```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Vấn đề: Port đã được sử dụng

**Giải pháp**:
```bash
# Tìm process đang dùng port
sudo lsof -i :8000

# Kill process
sudo kill -9 <PID>
```

---

## 📊 Monitoring và Maintenance

### Xem logs

```bash
# Logs của tất cả services
docker-compose logs -f

# Logs của service cụ thể
docker-compose logs -f backend
```

### Backup EBS Volume

```bash
# Tạo snapshot từ AWS Console
# Hoặc dùng AWS CLI
aws ec2 create-snapshot --volume-id vol-xxxxx --description "Backup $(date +%Y-%m-%d)"
```

### Update ứng dụng

```bash
# Nếu dùng CI/CD: chỉ cần push code
git push origin main

# Nếu deploy thủ công:
git pull origin main
../scripts/deploy.sh
```

---

## 💰 Tối ưu chi phí

1. **Sử dụng Reserved Instances**: Giảm 30-50% chi phí nếu commit 1-3 năm
2. **Spot Instances**: Giảm 70-90% nhưng có thể bị terminate
3. **EBS gp3**: Rẻ hơn gp2, đủ cho hầu hết use cases
4. **Auto Scaling**: Tự động scale down khi không dùng
5. **CloudWatch Alarms**: Monitor và alert khi có vấn đề

---

## 🔒 Security Best Practices

1. **Không commit `.env` file**: Thêm vào `.gitignore`
2. **Sử dụng Security Groups**: Chỉ mở ports cần thiết
3. **Rotate SSH keys**: Định kỳ thay đổi keys
4. **Enable CloudWatch Logs**: Monitor access logs
5. **Backup định kỳ**: Tạo snapshot EBS hàng tuần
6. **Update system**: `sudo apt update && sudo apt upgrade` định kỳ

---

## 📞 Support

Nếu gặp vấn đề, kiểm tra:
1. Logs: `docker-compose logs`
2. EC2 System Logs: AWS Console → EC2 → Instances → Actions → Monitor and troubleshoot → Get system log
3. CloudWatch: Xem metrics và logs

---

**Chúc bạn deploy thành công! 🎉**

