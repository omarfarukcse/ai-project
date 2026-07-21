# infrastructure/terraform/s3.tf
# S3 Storage Configuration

# ============================================================================
# 📊 S3 Bucket - Data Storage
# ============================================================================
resource "aws_s3_bucket" "data" {
  bucket = "${var.project_name}-${var.environment}-data-${data.aws_caller_identity.current.account_id}"
  
  tags = {
    Name        = "${var.project_name}-data-bucket"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  
  rule {
    id     = "archive-old-data"
    status = "Enabled"
    
    filter {
      prefix = "processed/"
    }
    
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    
    expiration {
      days = 365
    }
  }
  
  rule {
    id     = "expire-temp-data"
    status = "Enabled"
    
    filter {
      prefix = "temp/"
    }
    
    expiration {
      days = 7
    }
  }
}

# ============================================================================
# 📊 S3 Bucket - Model Storage
# ============================================================================
resource "aws_s3_bucket" "models" {
  bucket = "${var.project_name}-${var.environment}-models-${data.aws_caller_identity.current.account_id}"
  
  tags = {
    Name        = "${var.project_name}-models-bucket"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  
  rule {
    id     = "archive-old-models"
    status = "Enabled"
    
    filter {
      prefix = "archived/"
    }
    
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    
    expiration {
      days = 365
    }
  }
}

# ============================================================================
# 📊 S3 Bucket - Logs
# ============================================================================
resource "aws_s3_bucket" "logs" {
  bucket = "${var.project_name}-${var.environment}-logs-${data.aws_caller_identity.current.account_id}"
  
  tags = {
    Name        = "${var.project_name}-logs-bucket"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "logs" {
  bucket = aws_s3_bucket.logs.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "logs" {
  bucket = aws_s3_bucket.logs.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id
  
  rule {
    id     = "expire-old-logs"
    status = "Enabled"
    
    expiration {
      days = 90
    }
  }
}

# ============================================================================
# 📊 S3 Bucket - MLflow Artifacts
# ============================================================================
resource "aws_s3_bucket" "mlflow" {
  bucket = "${var.project_name}-${var.environment}-mlflow-${data.aws_caller_identity.current.account_id}"
  
  tags = {
    Name        = "${var.project_name}-mlflow-bucket"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "mlflow" {
  bucket = aws_s3_bucket.mlflow.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow" {
  bucket = aws_s3_bucket.mlflow.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "mlflow" {
  bucket = aws_s3_bucket.mlflow.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ============================================================================
# 📊 S3 Bucket Policies
# ============================================================================
resource "aws_s3_bucket_policy" "data" {
  bucket = aws_s3_bucket.data.id
  policy = data.aws_iam_policy_document.s3_data_policy.json
}

data "aws_iam_policy_document" "s3_data_policy" {
  statement {
    effect = "Deny"
    
    principals {
      type        = "AWS"
      identifiers = ["*"]
    }
    
    actions = [
      "s3:*"
    ]
    
    resources = [
      aws_s3_bucket.data.arn,
      "${aws_s3_bucket.data.arn}/*"
    ]
    
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}