# infrastructure/terraform/variables.tf
# Input Variables for CDSS Infrastructure

# ============================================================================
# 🏷️ General Variables
# ============================================================================
variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "production", "test"], var.environment)
    error_message = "Environment must be dev, staging, production, or test."
  }
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "cdss-healthcare"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "AWS profile name"
  type        = string
  default     = "default"
}

variable "cost_center" {
  description = "Cost center for billing"
  type        = string
  default     = "AI-Healthcare"
}

variable "team_name" {
  description = "Team name"
  type        = string
  default     = "ML-Platform"
}

# ============================================================================
# 🖥️ Compute Variables
# ============================================================================
variable "instance_type" {
  description = "EC2 instance type for API servers"
  type        = string
  default     = "t3.medium"
}

variable "instance_type_worker" {
  description = "EC2 instance type for Celery workers"
  type        = string
  default     = "t3.large"
}

variable "instance_type_gpu" {
  description = "EC2 instance type for GPU workloads"
  type        = string
  default     = "g4dn.xlarge"
}

variable "key_name" {
  description = "EC2 key pair name"
  type        = string
  default     = "cdss-key"
}

variable "min_instances" {
  description = "Minimum number of instances"
  type        = number
  default     = 2
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
  default     = 10
}

variable "desired_instances" {
  description = "Desired number of instances"
  type        = number
  default     = 3
}

# ============================================================================
# 💾 Storage Variables
# ============================================================================
variable "db_engine" {
  description = "RDS database engine"
  type        = string
  default     = "postgres"
}

variable "db_engine_version" {
  description = "RDS database engine version"
  type        = string
  default     = "15.3"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "cdss"
}

variable "db_username" {
  description = "Database username"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "db_allocated_storage" {
  description = "Database allocated storage in GB"
  type        = number
  default     = 100
}

variable "db_max_allocated_storage" {
  description = "Maximum database storage in GB"
  type        = number
  default     = 1000
}

variable "db_retention_period" {
  description = "Database backup retention period in days"
  type        = number
  default     = 30
}

# ============================================================================
# 🌐 Networking Variables
# ============================================================================
variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

# ============================================================================
# 🔐 Security Variables
# ============================================================================
variable "allowed_cidr_blocks" {
  description = "Allowed CIDR blocks for SSH and HTTPS"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict in production
}

variable "enable_encryption" {
  description = "Enable encryption for all services"
  type        = bool
  default     = true
}

variable "enable_backup" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

# ============================================================================
# 📊 Monitoring Variables
# ============================================================================
variable "enable_monitoring" {
  description = "Enable CloudWatch monitoring"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

variable "enable_alarms" {
  description = "Enable CloudWatch alarms"
  type        = bool
  default     = true
}

# ============================================================================
# 🚀 EKS Variables
# ============================================================================
variable "eks_version" {
  description = "EKS cluster version"
  type        = string
  default     = "1.27"
}

variable "eks_node_group_instance_types" {
  description = "EKS node group instance types"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
}

variable "eks_desired_size" {
  description = "EKS node group desired size"
  type        = number
  default     = 3
}

variable "eks_min_size" {
  description = "EKS node group minimum size"
  type        = number
  default     = 2
}

variable "eks_max_size" {
  description = "EKS node group maximum size"
  type        = number
  default     = 10
}

# ============================================================================
# 🔄 Auto Scaling Variables
# ============================================================================
variable "asg_min_size" {
  description = "Auto Scaling Group minimum size"
  type        = number
  default     = 2
}

variable "asg_max_size" {
  description = "Auto Scaling Group maximum size"
  type        = number
  default     = 10
}

variable "asg_desired_capacity" {
  description = "Auto Scaling Group desired capacity"
  type        = number
  default     = 3
}

variable "asg_health_check_grace_period" {
  description = "Auto Scaling Group health check grace period in seconds"
  type        = number
  default     = 300
}

# ============================================================================
# 🏷️ Tags
# ============================================================================
variable "additional_tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}