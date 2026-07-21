# infrastructure/terraform/security.tf
# Security Groups and Rules

# ============================================================================
# 📊 Security Group - API
# ============================================================================
resource "aws_security_group" "api" {
  name        = "${var.project_name}-api-sg-${var.environment}"
  description = "Security group for CDSS API servers"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    description = "HTTPS from ALB"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  
  ingress {
    description = "HTTP from ALB"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  
  ingress {
    description = "Application port"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  
  ingress {
    description = "SSH from bastion"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    security_groups = [aws_security_group.bastion.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "${var.project_name}-api-sg-${var.environment}"
    Environment = var.environment
  }
}

# ============================================================================
# 📊 Security Group - Database
# ============================================================================
resource "aws_security_group" "database" {
  name        = "${var.project_name}-db-sg-${var.environment}"
  description = "Security group for RDS database"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    description     = "PostgreSQL from API"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.api.id, aws_security_group.worker.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "${var.project_name}-db-sg-${var.environment}"
    Environment = var.environment
  }
}

# ============================================================================
# 📊 Security Group - Redis
# ============================================================================
resource "aws_security_group" "redis" {
  name        = "${var.project_name}-redis-sg-${var.environment}"
  description = "Security group for Redis cache"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    description     = "Redis from API"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.api.id, aws_security_group.worker.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "${var.project_name}-redis-sg-${var.environment}"
    Environment = var.environment
  }
}

# ============================================================================
# 📊 Security Group - Load Balancer
# ============================================================================
resource "aws_security_group" "alb" {
  name        = "${var.project_name}-alb-sg-${var.environment}"
  description = "Security group for Application Load Balancer"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    description = "HTTPS from internet"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }
  
  ingress {
    description = "HTTP from internet (redirect to HTTPS)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "${var.project_name}-alb-sg-${var.environment}"
    Environment = var.environment
  }
}

# ============================================================================
# 📊 Security Group - Bastion
# ============================================================================
resource "aws_security_group" "bastion" {
  name        = "${var.project_name}-bastion-sg-${var.environment}"
  description = "Security group for bastion host"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    description = "SSH from allowed CIDR"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "${var.project_name}-bastion-sg-${var.environment}"
    Environment = var.environment
  }
}

# ============================================================================
# 📊 Security Group - Worker
# ============================================================================
resource "aws_security_group" "worker" {
  name        = "${var.project_name}-worker-sg-${var.environment}"
  description = "Security group for Celery workers"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    description     = "Application port"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.api.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "${var.project_name}-worker-sg-${var.environment}"
    Environment = var.environment
  }
}