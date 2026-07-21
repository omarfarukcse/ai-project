# infrastructure/terraform/ec2.tf
# EC2 Compute Resources

# ============================================================================
# 📊 Launch Template - API
# ============================================================================
resource "aws_launch_template" "api" {
  name_prefix   = "${var.project_name}-api-${var.environment}-"
  image_id      = data.aws_ami.amazon_linux_2.id
  instance_type = var.instance_type
  key_name      = var.key_name
  
  vpc_security_group_ids = [
    aws_security_group.api.id,
    aws_security_group.bastion.id
  ]
  
  user_data = base64encode(templatefile("${path.module}/user_data_api.sh", {
    environment = var.environment
    region      = var.aws_region
    db_host     = aws_db_instance.main.address
    db_name     = var.db_name
    db_user     = var.db_username
    db_password = var.db_password
    redis_host  = aws_elasticache_cluster.redis.cache_nodes[0].address
  }))
  
  iam_instance_profile {
    name = aws_iam_instance_profile.ec2.name
  }
  
  monitoring {
    enabled = var.enable_monitoring
  }
  
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }
  
  block_device_mappings {
    device_name = "/dev/xvda"
    
    ebs {
      volume_size           = 50
      volume_type           = "gp3"
      delete_on_termination = true
      encrypted             = var.enable_encryption
    }
  }
  
  tag_specifications {
    resource_type = "instance"
    
    tags = {
      Name        = "${var.project_name}-api-${var.environment}"
      Environment = var.environment
      Role        = "api"
    }
  }
}

# ============================================================================
# 📊 Launch Template - Worker
# ============================================================================
resource "aws_launch_template" "worker" {
  name_prefix   = "${var.project_name}-worker-${var.environment}-"
  image_id      = data.aws_ami.amazon_linux_2.id
  instance_type = var.instance_type_worker
  key_name      = var.key_name
  
  vpc_security_group_ids = [
    aws_security_group.worker.id,
    aws_security_group.bastion.id
  ]
  
  user_data = base64encode(templatefile("${path.module}/user_data_worker.sh", {
    environment = var.environment
    region      = var.aws_region
    db_host     = aws_db_instance.main.address
    db_name     = var.db_name
    db_user     = var.db_username
    db_password = var.db_password
    redis_host  = aws_elasticache_cluster.redis.cache_nodes[0].address
  }))
  
  iam_instance_profile {
    name = aws_iam_instance_profile.ec2.name
  }
  
  monitoring {
    enabled = var.enable_monitoring
  }
  
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }
  
  block_device_mappings {
    device_name = "/dev/xvda"
    
    ebs {
      volume_size           = 50
      volume_type           = "gp3"
      delete_on_termination = true
      encrypted             = var.enable_encryption
    }
  }
  
  tag_specifications {
    resource_type = "instance"
    
    tags = {
      Name        = "${var.project_name}-worker-${var.environment}"
      Environment = var.environment
      Role        = "worker"
    }
  }
}

# ============================================================================
# 📊 Auto Scaling Group - API
# ============================================================================
resource "aws_autoscaling_group" "api" {
  name = "${var.project_name}-api-asg-${var.environment}"
  
  min_size         = var.asg_min_size
  max_size         = var.asg_max_size
  desired_capacity = var.asg_desired_capacity
  
  vpc_zone_identifier = aws_subnet.private[*].id
  
  launch_template {
    id      = aws_launch_template.api.id
    version = "$Latest"
  }
  
  health_check_type         = "ELB"
  health_check_grace_period = var.asg_health_check_grace_period
  
  target_group_arns = [aws_lb_target_group.api.arn]
  
  enabled_metrics = [
    "GroupMinSize",
    "GroupMaxSize",
    "GroupDesiredCapacity",
    "GroupInServiceInstances",
    "GroupTotalInstances"
  ]
  
  tag {
    key                 = "Name"
    value               = "${var.project_name}-api-${var.environment}"
    propagate_at_launch = true
  }
  
  tag {
    key                 = "Environment"
    value               = var.environment
    propagate_at_launch = true
  }
  
  tag {
    key                 = "Role"
    value               = "api"
    propagate_at_launch = true
  }
}

# ============================================================================
# 📊 Auto Scaling Group - Worker
# ============================================================================
resource "aws_autoscaling_group" "worker" {
  name = "${var.project_name}-worker-asg-${var.environment}"
  
  min_size         = var.asg_min_size
  max_size         = var.asg_max_size
  desired_capacity = var.asg_desired_capacity
  
  vpc_zone_identifier = aws_subnet.private[*].id
  
  launch_template {
    id      = aws_launch_template.worker.id
    version = "$Latest"
  }
  
  health_check_type         = "EC2"
  health_check_grace_period = var.asg_health_check_grace_period
  
  enabled_metrics = [
    "GroupMinSize",
    "GroupMaxSize",
    "GroupDesiredCapacity",
    "GroupInServiceInstances",
    "GroupTotalInstances"
  ]
  
  tag {
    key                 = "Name"
    value               = "${var.project_name}-worker-${var.environment}"
    propagate_at_launch = true
  }
  
  tag {
    key                 = "Environment"
    value               = var.environment
    propagate_at_launch = true
  }
  
  tag {
    key                 = "Role"
    value               = "worker"
    propagate_at_launch = true
  }
}

# ============================================================================
# 📊 AMI Data Source
# ============================================================================
data "aws_ami" "amazon_linux_2" {
  most_recent = true
  owners      = ["amazon"]
  
  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
  
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}