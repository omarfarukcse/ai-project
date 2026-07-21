# infrastructure/terraform/eks.tf
# EKS Kubernetes Cluster Configuration

# ============================================================================
# 📊 EKS Cluster
# ============================================================================
resource "aws_eks_cluster" "main" {
  name     = "${var.project_name}-eks-${var.environment}"
  version  = var.eks_version
  role_arn = aws_iam_role.eks.arn
  
  vpc_config {
    subnet_ids = concat(aws_subnet.public[*].id, aws_subnet.private[*].id)
    security_group_ids = [
      aws_security_group.eks_cluster.id
    ]
    endpoint_private_access = true
    endpoint_public_access  = var.environment != "production"
  }
  
  enabled_cluster_log_types = [
    "api",
    "audit",
    "authenticator",
    "controllerManager",
    "scheduler"
  ]
  
  tags = {
    Name        = "${var.project_name}-eks-${var.environment}"
    Environment = var.environment
  }
}

# ============================================================================
# 📊 EKS Security Group
# ============================================================================
resource "aws_security_group" "eks_cluster" {
  name        = "${var.project_name}-eks-cluster-sg-${var.environment}"
  description = "Security group for EKS cluster"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    description = "HTTPS from ALB"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  
  ingress {
    description = "API server port"
    from_port   = 6443
    to_port     = 6443
    protocol    = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "${var.project_name}-eks-cluster-sg"
    Environment = var.environment
  }
}

# ============================================================================
# 📊 EKS Node Group
# ============================================================================
resource "aws_eks_node_group" "main" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${var.project_name}-nodes-${var.environment}"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = aws_subnet.private[*].id
  
  instance_types = var.eks_node_group_instance_types
  
  scaling_config {
    desired_size = var.eks_desired_size
    max_size     = var.eks_max_size
    min_size     = var.eks_min_size
  }
  
  update_config {
    max_unavailable = 1
  }
  
  launch_template {
    id      = aws_launch_template.eks_nodes.id
    version = "$Latest"
  }
  
  tags = {
    Name        = "${var.project_name}-eks-nodes"
    Environment = var.environment
  }
}

# ============================================================================
# 📊 IAM Role - EKS Nodes
# ============================================================================
resource "aws_iam_role" "eks_nodes" {
  name = "${var.project_name}-eks-nodes-role-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
  
  tags = {
    Name        = "${var.project_name}-eks-nodes-role"
    Environment = var.environment
  }
}

resource "aws_iam_role_policy_attachment" "eks_nodes_worker" {
  role       = aws_iam_role.eks_nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "eks_nodes_cni" {
  role       = aws_iam_role.eks_nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role_policy_attachment" "eks_nodes_ecr" {
  role       = aws_iam_role.eks_nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "eks_nodes_ssm" {
  role       = aws_iam_role.eks_nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# ============================================================================
# 📊 Launch Template - EKS Nodes
# ============================================================================
resource "aws_launch_template" "eks_nodes" {
  name_prefix   = "${var.project_name}-eks-nodes-${var.environment}-"
  image_id      = data.aws_ami.eks_optimized.id
  instance_type = var.eks_node_group_instance_types[0]
  
  vpc_security_group_ids = [
    aws_security_group.eks_nodes.id
  ]
  
  user_data = base64encode(templatefile("${path.module}/user_data_eks.sh", {
    cluster_name = aws_eks_cluster.main.name
    environment  = var.environment
  }))
  
  iam_instance_profile {
    name = aws_iam_instance_profile.eks_nodes.name
  }
  
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }
  
  block_device_mappings {
    device_name = "/dev/xvda"
    
    ebs {
      volume_size           = 100
      volume_type           = "gp3"
      delete_on_termination = true
      encrypted             = var.enable_encryption
    }
  }
  
  tag_specifications {
    resource_type = "instance"
    
    tags = {
      Name        = "${var.project_name}-eks-node-${var.environment}"
      Environment = var.environment
      Role        = "eks-node"
    }
  }
}

# ============================================================================
# 📊 IAM Instance Profile - EKS Nodes
# ============================================================================
resource "aws_iam_instance_profile" "eks_nodes" {
  name = "${var.project_name}-eks-nodes-profile-${var.environment}"
  role = aws_iam_role.eks_nodes.name
}

# ============================================================================
# 📊 Security Group - EKS Nodes
# ============================================================================
resource "aws_security_group" "eks_nodes" {
  name        = "${var.project_name}-eks-nodes-sg-${var.environment}"
  description = "Security group for EKS nodes"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    description     = "Node communication"
    from_port       = 0
    to_port         = 65535
    protocol        = "tcp"
    self            = true
  }
  
  ingress {
    description     = "Kubelet"
    from_port       = 10250
    to_port         = 10250
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "${var.project_name}-eks-nodes-sg"
    Environment = var.environment
  }
}

# ============================================================================
# 📊 AMI Data Source - EKS Optimized
# ============================================================================
data "aws_ami" "eks_optimized" {
  most_recent = true
  owners      = ["amazon"]
  
  filter {
    name   = "name"
    values = ["amazon-eks-node-${var.eks_version}-v*"]
  }
  
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}