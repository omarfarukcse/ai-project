# infrastructure/terraform/main.tf
# AWS Infrastructure for CDSS

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
  backend "s3" {
    bucket = "cdss-terraform-state"
    key    = "terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region"
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  default     = "production"
}

variable "project_name" {
  description = "Project name"
  default     = "cdss"
}

# ============================================================================
# VPC Configuration
# ============================================================================
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name        = "${var.project_name}-vpc"
    Environment = var.environment
  }
}

resource "aws_subnet" "public" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {
    Name        = "${var.project_name}-public-subnet-${count.index + 1}"
    Environment = var.environment
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}

# ============================================================================
# EKS Cluster
# ============================================================================
resource "aws_eks_cluster" "cdss" {
  name     = "${var.project_name}-cluster"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.27"
  
  vpc_config {
    subnet_ids = aws_subnet.public[*].id
  }
  
  tags = {
    Name        = "${var.project_name}-cluster"
    Environment = var.environment
  }
}

# ============================================================================
# IAM Roles
# ============================================================================
resource "aws_iam_role" "eks_cluster" {
  name = "${var.project_name}-eks-cluster-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "eks.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster.name
}

resource "aws_iam_role_policy_attachment" "eks_service_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSServicePolicy"
  role       = aws_iam_role.eks_cluster.name
}

# ============================================================================
# RDS (PostgreSQL for metadata)
# ============================================================================
resource "aws_db_instance" "cdss" {
  identifier           = "${var.project_name}-db"
  engine              = "postgres"
  engine_version      = "15.3"
  instance_class      = "db.t3.medium"
  allocated_storage   = 100
  storage_type        = "gp3"
  db_name             = "cdss"
  username            = var.db_username
  password            = var.db_password
  vpc_security_group_ids = [aws_security_group.db.id]
  db_subnet_group_name  = aws_db_subnet_group.cdss.name
  backup_retention_period = 30
  skip_final_snapshot    = false
  final_snapshot_identifier = "${var.project_name}-final-snapshot"
  
  tags = {
    Name        = "${var.project_name}-db"
    Environment = var.environment
  }
}

resource "aws_db_subnet_group" "cdss" {
  name       = "${var.project_name}-db-subnet-group"
  subnet_ids = aws_subnet.public[*].id
}

# ============================================================================
# Security Groups
# ============================================================================
resource "aws_security_group" "db" {
  name        = "${var.project_name}-db-sg"
  description = "RDS security group"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.main.cidr_block]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "${var.project_name}-db-sg"
    Environment = var.environment
  }
}