# infrastructure/terraform/alb.tf
# Application Load Balancer Configuration

# ============================================================================
# 📊 Application Load Balancer
# ============================================================================
resource "aws_lb" "main" {
  name               = "${var.project_name}-alb-${var.environment}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
  
  enable_deletion_protection = var.environment == "production"
  enable_http2               = true
  enable_cross_zone_loading  = true
  drop_invalid_header_fields = true
  
  access_logs {
    bucket  = aws_s3_bucket.logs.bucket
    prefix  = "alb-logs"
    enabled = var.environment == "production"
  }
  
  tags = {
    Name        = "${var.project_name}-alb-${var.environment}"
    Environment = var.environment
  }
}

# ============================================================================
# 📊 Target Group - API
# ============================================================================
resource "aws_lb_target_group" "api" {
  name        = "${var.project_name}-api-tg-${var.environment}"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "instance"
  
  health_check {
    enabled             = true
    healthy_threshold   = 3
    unhealthy_threshold = 3
    interval            = 30
    timeout             = 10
    path                = "/health"
    protocol            = "HTTP"
    matcher             = "200"
  }
  
  stickiness {
    enabled    = false
    type       = "lb_cookie"
    cookie_duration = 3600
  }
  
  tags = {
    Name        = "${var.project_name}-api-tg"
    Environment = var.environment
  }
}

# ============================================================================
# 📊 Target Group - Worker
# ============================================================================
resource "aws_lb_target_group" "worker" {
  name        = "${var.project_name}-worker-tg-${var.environment}"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "instance"
  
  health_check {
    enabled             = true
    healthy_threshold   = 3
    unhealthy_threshold = 3
    interval            = 30
    timeout             = 10
    path                = "/health"
    protocol            = "HTTP"
    matcher             = "200"
  }
  
  tags = {
    Name        = "${var.project_name}-worker-tg"
    Environment = var.environment
  }
}

# ============================================================================
# 📊 Target Group - Dashboard
# ============================================================================
resource "aws_lb_target_group" "dashboard" {
  name        = "${var.project_name}-dashboard-tg-${var.environment}"
  port        = 8501
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "instance"
  
  health_check {
    enabled             = true
    healthy_threshold   = 3
    unhealthy_threshold = 3
    interval            = 30
    timeout             = 10
    path                = "/_stcore/health"
    protocol            = "HTTP"
    matcher             = "200"
  }
  
  tags = {
    Name        = "${var.project_name}-dashboard-tg"
    Environment = var.environment
  }
}

# ============================================================================
# 📊 HTTP Listener
# ============================================================================
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"
  
  default_action {
    type = "redirect"
    
    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

# ============================================================================
# 📊 HTTPS Listener
# ============================================================================
resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.main.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-2016-08"
  certificate_arn   = aws_acm_certificate.main.arn
  
  default_action {
    type = "fixed-response"
    
    fixed_response {
      content_type = "application/json"
      message_body = "{\"error\":\"No route found\"}"
      status_code  = "404"
    }
  }
}

# ============================================================================
# 📊 HTTPS Listener Rules
# ============================================================================
resource "aws_lb_listener_rule" "api" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 10
  
  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
  
  condition {
    host_header {
      values = ["api.${var.domain_name}"]
    }
  }
}

resource "aws_lb_listener_rule" "dashboard" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 20
  
  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.dashboard.arn
  }
  
  condition {
    host_header {
      values = ["dashboard.${var.domain_name}"]
    }
  }
}

# ============================================================================
# 📊 SSL Certificate (ACM)
# ============================================================================
resource "aws_acm_certificate" "main" {
  domain_name       = var.domain_name
  subject_alternative_names = ["*.${var.domain_name}"]
  validation_method = "DNS"
  
  lifecycle {
    create_before_destroy = true
  }
  
  tags = {
    Name        = "${var.project_name}-cert"
    Environment = var.environment
  }
}
