# src/security/__init__.py
"""
Security Package - Enterprise Security Layer

This package provides production-grade security with:
- JWT Authentication: Token-based authentication with refresh tokens
- Rate Limiting: Distributed rate limiting with token bucket
- Data Encryption: AES-256 encryption for sensitive data
- Security Headers: OWASP recommended security headers
- Audit Logging: Security event tracking
- API Key Management: Secure API key handling
- CORS Configuration: Cross-origin resource sharing

Architecture:
    auth.py         → JWT authentication and authorization
    rate_limiter.py → Distributed rate limiting
    encryption.py   → Data encryption and decryption

Features:
    - JWT with refresh tokens
    - Role-based access control (RBAC)
    - Distributed rate limiting (Redis)
    - AES-256-GCM encryption
    - Security event logging
    - Password hashing (bcrypt)
    - API key validation

Version: 3.0.0
"""

from src.security.auth import (
    AuthManager,
    JWTHandler,
    PasswordHasher,
    TokenValidator,
    PermissionChecker,
    get_auth_manager,
    authenticate_user,
    require_permission,
    require_roles,
    TokenType,
    TokenData,
    User,
)
from src.security.rate_limiter import (
    RateLimiter,
    TokenBucket,
    DistributedRateLimiter,
    RateLimitConfig,
    RateLimitExceeded,
    get_rate_limiter,
)
from src.security.encryption import (
    EncryptionManager,
    AESCipher,
    KeyManager,
    SecureStorage,
    DataMasker,
    get_encryption_manager,
)

__version__ = "3.0.0"
__all__ = [
    # Auth
    "AuthManager",
    "JWTHandler",
    "PasswordHasher",
    "TokenValidator",
    "PermissionChecker",
    "get_auth_manager",
    "authenticate_user",
    "require_permission",
    "require_roles",
    "TokenType",
    "TokenData",
    "User",
    
    # Rate Limiter
    "RateLimiter",
    "TokenBucket",
    "DistributedRateLimiter",
    "RateLimitConfig",
    "RateLimitExceeded",
    "get_rate_limiter",
    
    # Encryption
    "EncryptionManager",
    "AESCipher",
    "KeyManager",
    "SecureStorage",
    "DataMasker",
    "get_encryption_manager",
]

import logging
logger = logging.getLogger(__name__)
logger.info(f"🔒 Security Package v{__version__} initialized")