# infrastructure/terraform/outputs.tf
# Terraform Outputs

# ============================================================================
# 📊 Networking Outputs
# ============================================================================
output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = aws_subnet.private[*].id
}

# ============================================================================
# 📊 Database Outputs
# ============================================================================
output "db_endpoint" {
  description = "RDS database endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = true
}

output "db_address" {
  description = "RDS database address"
  value       = aws_db_instance.main.address
}

output "db_port" {
  description = "RDS database port"
  value       = aws_db_instance.main.port
}

# ============================================================================
# 📊 Load Balancer Outputs
# ============================================================================
output "alb_dns_name" {
  description = "ALB DNS name"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "ALB zone ID"
  value       = aws_lb.main.zone_id
}

output "alb_arn" {
  description = "ALB ARN"
  value       = aws_lb.main.arn
}

# ============================================================================
# 📊 S3 Outputs
# ============================================================================
output "s3_data_bucket" {
  description = "S3 data bucket name"
  value       = aws_s3_bucket.data.id
}

output "s3_models_bucket" {
  description = "S3 models bucket name"
  value       = aws_s3_bucket.models.id
}

output "s3_logs_bucket" {
  description = "S3 logs bucket name"
  value       = aws_s3_bucket.logs.id
}

# ============================================================================
# 📊 EKS Outputs
# ============================================================================
output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.main.name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = aws_eks_cluster.main.endpoint
}

output "eks_cluster_security_group_id" {
  description = "EKS cluster security group ID"
  value       = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
}

# ============================================================================
# 📊 Secrets Outputs
# ============================================================================
output "db_secret_arn" {
  description = "RDS credentials secret ARN"
  value       = aws_secretsmanager_secret.db_credentials.arn
}

# ============================================================================
# 📊 Monitoring Outputs
# ============================================================================
output "sns_topic_arn" {
  description = "SNS topic ARN for alerts"
  value       = aws_sns_topic.alerts.arn
}

# ============================================================================
# 📊 Auto Scaling Outputs
# ============================================================================
output "api_asg_name" {
  description = "API auto scaling group name"
  value       = aws_autoscaling_group.api.name
}

output "worker_asg_name" {
  description = "Worker auto scaling group name"
  value       = aws_autoscaling_group.worker.name
}
