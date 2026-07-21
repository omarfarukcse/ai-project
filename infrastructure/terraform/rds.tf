# infrastructure/terraform/rds.tf
# RDS Database Configuration

# ============================================================================
# 📊 RDS Subnet Group
# ============================================================================
resource "aws_db_subnet_group" "main" {
  name        = "${var.project_name}-db-subnet-group-${var.environment}"
  description = "RDS subnet group for CDSS"
  subnet_ids  = aws_subnet.private[*].id
  
  tags = {
    Name        = "${var.project_name}-db-subnet-group"
    Environment = var.environment
  }
}

# ============================================================================
# 📊 RDS Parameter Group
# ============================================================================
resource "aws_db_parameter_group" "main" {
  name        = "${var.project_name}-db-params-${var.environment}"
  family      = "postgres15"
  description = "RDS parameter group for CDSS"
  
  parameter {
    name  = "shared_buffers"
    value = "256MB"
  }
  
  parameter {
    name  = "effective_cache_size"
    value = "768MB"
  }
  
  parameter {
    name  = "work_mem"
    value = "16MB"
  }
  
  parameter {
    name  = "maintenance_work_mem"
    value = "64MB"
  }
  
  parameter {
    name  = "max_connections"
    value = "200"
  }
  
  parameter {
    name  = "log_min_duration_statement"
    value = "1000"
  }
  
  parameter {
    name  = "log_statement"
    value = "ddl"
  }
  
  parameter {
    name  = "log_checkpoints"
    value = "on"
  }
  
  parameter {
    name  = "log_lock_waits"
    value = "on"
  }
  
  parameter {
    name  = "log_temp_files"
    value = "1024"
  }
  
  tags = {
    Name        = "${var.project_name}-db-params"
    Environment = var.environment
  }
}

# ============================================================================
# 📊 RDS Database Instance
# ============================================================================
resource "aws_db_instance" "main" {
  identifier = "${var.project_name}-db-${var.environment}"
  
  engine               = var.db_engine
  engine_version       = var.db_engine_version
  instance_class       = var.db_instance_class
  allocated_storage    = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_type         = "gp3"
  storage_encrypted    = var.enable_encryption
  
  db_name  = var.db_name
  username = var.db_username
  password = var.db_password
  
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.database.id]
  parameter_group_name   = aws_db_parameter_group.main.name
  
  backup_retention_period = var.db_retention_period
  backup_window          = "03:00-04:00"
  maintenance_window     = "Sun:04:00-Sun:05:00"
  
  auto_minor_version_upgrade = true
  copy_tags_to_snapshot      = true
  deletion_protection        = var.environment == "production"
  skip_final_snapshot        = var.environment != "production"
  final_snapshot_identifier  = "${var.project_name}-final-snapshot-${var.environment}-${formatdate("YYYY-MM-DD-hh-mm", timestamp())}"
  
  enabled_cloudwatch_logs_exports = [
    "postgresql",
    "upgrade"
  ]
  
  tags = {
    Name        = "${var.project_name}-db-${var.environment}"
    Environment = var.environment
  }
}

# ============================================================================
# 📊 RDS Read Replica (Optional)
# ============================================================================
resource "aws_db_instance" "replica" {
  count = var.environment == "production" ? 1 : 0
  
  identifier = "${var.project_name}-db-replica-${var.environment}"
  
  replicate_source_db = aws_db_instance.main.identifier
  instance_class      = var.db_instance_class
  storage_encrypted   = var.enable_encryption
  
  vpc_security_group_ids = [aws_security_group.database.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "04:00-05:00"
  maintenance_window     = "Sun:05:00-Sun:06:00"
  
  auto_minor_version_upgrade = true
  copy_tags_to_snapshot      = true
  deletion_protection        = var.environment == "production"
  skip_final_snapshot        = true
  
  tags = {
    Name        = "${var.project_name}-db-replica-${var.environment}"
    Environment = var.environment
    Role        = "replica"
  }
}

# ============================================================================
# 📊 RDS Performance Insights
# ============================================================================
resource "aws_db_instance" "main" {
  # ... existing configuration ...
  
  performance_insights_enabled = true
  performance_insights_retention_period = 7
  
  # Added to existing resource
}

# ============================================================================
# 📊 Secrets Manager for RDS Credentials
# ============================================================================
resource "aws_secretsmanager_secret" "db_credentials" {
  name        = "${var.project_name}/db/${var.environment}/credentials"
  description = "RDS database credentials for CDSS"
  
  rotation_rules {
    automatically_after_days = 30
  }
  
  tags = {
    Name        = "${var.project_name}-db-secret"
    Environment = var.environment
  }
}

resource "aws_secretsmanager_secret_version" "db_credentials" {
  secret_id = aws_secretsmanager_secret.db_credentials.id
  
  secret_string = jsonencode({
    username = var.db_username
    password = var.db_password
    host     = aws_db_instance.main.address
    port     = aws_db_instance.main.port
    dbname   = var.db_name
  })
}

# ============================================================================
# 📊 CloudWatch Alarm - Database
# ============================================================================
resource "aws_cloudwatch_metric_alarm" "db_cpu" {
  alarm_name          = "${var.project_name}-db-cpu-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "Database CPU utilization is high"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.id
  }
  
  tags = {
    Name        = "${var.project_name}-db-cpu-alarm"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_metric_alarm" "db_storage" {
  alarm_name          = "${var.project_name}-db-storage-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "FreeStorageSpace"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "10737418240"  # 10GB
  alarm_description   = "Database storage space is low"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.id
  }
  
  tags = {
    Name        = "${var.project_name}-db-storage-alarm"
    Environment = var.environment
  }
}