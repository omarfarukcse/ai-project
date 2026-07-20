# src/security/auth.py
"""
JWT Authentication and Authorization with RBAC
"""

import os
import jwt
import bcrypt
import uuid
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from functools import wraps
import asyncio

from src.logger import get_logger
from src.config_manager import config_manager
from src.caching.redis_client import get_fast_redis_client

logger = get_logger(__name__)


class TokenType(Enum):
    """Token types"""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"


@dataclass
class TokenData:
    """Token payload data"""
    user_id: str
    username: str
    roles: List[str]
    permissions: List[str]
    token_type: TokenType = TokenType.ACCESS
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=1))
    issued_at: datetime = field(default_factory=datetime.now)
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "roles": self.roles,
            "permissions": self.permissions,
            "token_type": self.token_type.value,
            "expires_at": self.expires_at.isoformat(),
            "issued_at": self.issued_at.isoformat(),
            "additional_data": self.additional_data,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TokenData':
        return cls(
            user_id=data["user_id"],
            username=data["username"],
            roles=data.get("roles", []),
            permissions=data.get("permissions", []),
            token_type=TokenType(data.get("token_type", "access")),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            issued_at=datetime.fromisoformat(data["issued_at"]),
            additional_data=data.get("additional_data", {}),
        )


@dataclass
class User:
    """User data"""
    id: str
    username: str
    email: str
    password_hash: str
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "roles": self.roles,
            "permissions": self.permissions,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "additional_data": self.additional_data,
        }


class JWTHandler:
    """
    JWT Token Handler with:
    - Token generation
    - Token validation
    - Token refresh
    - Token revocation
    """
    
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")
        self.algorithm = "HS256"
        self.access_token_expiry = timedelta(hours=1)
        self.refresh_token_expiry = timedelta(days=7)
        
        logger.info("🔑 JWTHandler initialized")
    
    def generate_token(
        self,
        user_id: str,
        username: str,
        roles: List[str],
        permissions: List[str],
        token_type: TokenType = TokenType.ACCESS,
        expires_in: Optional[timedelta] = None,
    ) -> Tuple[str, datetime]:
        """Generate a JWT token"""
        
        # Determine expiry
        if expires_in:
            expiry = expires_in
        elif token_type == TokenType.ACCESS:
            expiry = self.access_token_expiry
        elif token_type == TokenType.REFRESH:
            expiry = self.refresh_token_expiry
        else:
            expiry = timedelta(days=365)  # API Key
        
        now = datetime.now()
        expires_at = now + expiry
        
        # Prepare payload
        payload = {
            "user_id": user_id,
            "username": username,
            "roles": roles,
            "permissions": permissions,
            "token_type": token_type.value,
            "exp": int(expires_at.timestamp()),
            "iat": int(now.timestamp()),
            "jti": str(uuid.uuid4()),
        }
        
        # Sign token
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        return token, expires_at
    
    def validate_token(self, token: str) -> Optional[TokenData]:
        """Validate and decode a JWT token"""
        
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_signature": True, "verify_exp": True},
            )
            
            # Check if token is revoked
            # This would check against a token blacklist
            
            return TokenData(
                user_id=payload["user_id"],
                username=payload["username"],
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                token_type=TokenType(payload.get("token_type", "access")),
                expires_at=datetime.fromtimestamp(payload["exp"]),
                issued_at=datetime.fromtimestamp(payload["iat"]),
                additional_data=payload.get("additional_data", {}),
            )
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Token validation error: {str(e)}")
            return None
    
    def refresh_token(self, refresh_token: str) -> Tuple[Optional[str], Optional[datetime]]:
        """Refresh an access token using a refresh token"""
        
        # Validate refresh token
        token_data = self.validate_token(refresh_token)
        
        if not token_data or token_data.token_type != TokenType.REFRESH:
            logger.warning("Invalid refresh token")
            return None, None
        
        # Generate new access token
        new_token, expires_at = self.generate_token(
            user_id=token_data.user_id,
            username=token_data.username,
            roles=token_data.roles,
            permissions=token_data.permissions,
            token_type=TokenType.ACCESS,
        )
        
        return new_token, expires_at


class PasswordHasher:
    """
    Password hashing and verification with bcrypt
    """
    
    def __init__(self):
        self.rounds = 12
        
        logger.info("🔐 PasswordHasher initialized")
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        
        salt = bcrypt.gensalt(rounds=self.rounds)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash"""
        
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'),
                hashed.encode('utf-8')
            )
        except Exception:
            return False
    
    def needs_rehash(self, hashed: str) -> bool:
        """Check if password needs rehashing"""
        
        try:
            # Check if hash uses current cost
            return False  # Simple implementation
        except Exception:
            return True


class AuthManager:
    """
    Comprehensive Authentication Manager with:
    - JWT authentication
    - RBAC (Role-Based Access Control)
    - User management
    - Token blacklisting
    - Session management
    - Audit logging
    """
    
    def __init__(self):
        self.jwt_handler = JWTHandler()
        self.password_hasher = PasswordHasher()
        self.redis_client = get_fast_redis_client()
        self._blacklist_key_prefix = "token:blacklist:"
        self._user_cache_key_prefix = "user:cache:"
        self._users: Dict[str, User] = {}
        
        # Initialize default admin user
        self._init_default_users()
        
        logger.info("🔐 AuthManager initialized")
    
    def _init_default_users(self):
        """Initialize default users"""
        
        # Admin user
        admin = User(
            id="admin_001",
            username="admin",
            email="admin@cdss-healthcare.com",
            password_hash=self.password_hasher.hash_password("Admin123!"),
            roles=["admin", "ml_engineer", "clinician"],
            permissions=["*"],
        )
        self._users[admin.id] = admin
        self._users[admin.username] = admin
        
        # ML Engineer
        ml_engineer = User(
            id="ml_001",
            username="ml_engineer",
            email="ml@cdss-healthcare.com",
            password_hash=self.password_hasher.hash_password("MlEngineer123!"),
            roles=["ml_engineer"],
            permissions=["train", "evaluate", "deploy", "monitor"],
        )
        self._users[ml_engineer.id] = ml_engineer
        self._users[ml_engineer.username] = ml_engineer
        
        # Clinician
        clinician = User(
            id="clinician_001",
            username="clinician",
            email="clinician@cdss-healthcare.com",
            password_hash=self.password_hasher.hash_password("Clinician123!"),
            roles=["clinician"],
            permissions=["predict", "view_reports"],
        )
        self._users[clinician.id] = clinician
        self._users[clinician.username] = clinician
        
        logger.info("✅ Default users initialized")
    
    # ============================================================================
    # 🚀 Authentication Methods
    # ============================================================================
    
    async def authenticate(
        self,
        username: str,
        password: str,
    ) -> Optional[Tuple[str, str, datetime]]:
        """
        Authenticate a user and generate tokens
        
        Returns:
            Tuple of (access_token, refresh_token, expires_at)
        """
        
        # Find user
        user = self._get_user_by_username(username)
        if not user:
            logger.warning(f"Authentication failed: User {username} not found")
            return None
        
        # Check if enabled
        if not user.enabled:
            logger.warning(f"Authentication failed: User {username} disabled")
            return None
        
        # Verify password
        if not self.password_hasher.verify_password(password, user.password_hash):
            logger.warning(f"Authentication failed: Invalid password for {username}")
            return None
        
        # Update last login
        user.last_login = datetime.now()
        
        # Generate tokens
        access_token, access_expiry = self.jwt_handler.generate_token(
            user_id=user.id,
            username=user.username,
            roles=user.roles,
            permissions=user.permissions,
            token_type=TokenType.ACCESS,
        )
        
        refresh_token, refresh_expiry = self.jwt_handler.generate_token(
            user_id=user.id,
            username=user.username,
            roles=user.roles,
            permissions=user.permissions,
            token_type=TokenType.REFRESH,
        )
        
        # Cache user
        await self._cache_user(user)
        
        logger.info(f"✅ User authenticated: {username}")
        
        return access_token, refresh_token, access_expiry
    
    async def refresh_tokens(
        self,
        refresh_token: str,
    ) -> Optional[Tuple[str, str, datetime]]:
        """Refresh access token using refresh token"""
        
        # Validate refresh token
        token_data = self.jwt_handler.validate_token(refresh_token)
        if not token_data or token_data.token_type != TokenType.REFRESH:
            logger.warning("Invalid refresh token")
            return None
        
        # Check if token is blacklisted
        if await self._is_token_blacklisted(refresh_token):
            logger.warning("Refresh token is blacklisted")
            return None
        
        # Get user
        user = await self._get_cached_user(token_data.user_id)
        if not user:
            user = self._get_user_by_id(token_data.user_id)
            if not user:
                return None
        
        # Generate new tokens
        access_token, access_expiry = self.jwt_handler.generate_token(
            user_id=user.id,
            username=user.username,
            roles=user.roles,
            permissions=user.permissions,
            token_type=TokenType.ACCESS,
        )
        
        # Blacklist old refresh token
        await self._blacklist_token(refresh_token)
        
        logger.info(f"✅ Tokens refreshed for: {user.username}")
        
        return access_token, refresh_token, access_expiry
    
    async def validate_token(self, token: str) -> Optional[TokenData]:
        """Validate a token"""
        
        # Check if token is blacklisted
        if await self._is_token_blacklisted(token):
            logger.warning("Token is blacklisted")
            return None
        
        # Validate token
        token_data = self.jwt_handler.validate_token(token)
        if not token_data:
            return None
        
        # Check if user exists
        user = await self._get_cached_user(token_data.user_id)
        if not user:
            user = self._get_user_by_id(token_data.user_id)
            if not user:
                return None
        
        # Check if user is enabled
        if not user.enabled:
            return None
        
        return token_data
    
    async def revoke_token(self, token: str):
        """Revoke a token (add to blacklist)"""
        
        # Validate token to get expiry
        token_data = self.jwt_handler.validate_token(token)
        if token_data:
            # Add to blacklist until token expires
            await self._blacklist_token(token, token_data.expires_at)
            logger.info(f"🔴 Token revoked")
    
    # ============================================================================
    # 🔧 Authorization Methods
    # ============================================================================
    
    def has_permission(self, token_data: TokenData, required_permission: str) -> bool:
        """Check if user has a specific permission"""
        
        # Admin has all permissions
        if "admin" in token_data.roles:
            return True
        
        # Check permissions
        if "*" in token_data.permissions:
            return True
        
        return required_permission in token_data.permissions
    
    def has_role(self, token_data: TokenData, required_role: str) -> bool:
        """Check if user has a specific role"""
        
        return required_role in token_data.roles or "admin" in token_data.roles
    
    def has_any_role(self, token_data: TokenData, required_roles: List[str]) -> bool:
        """Check if user has any of the required roles"""
        
        if "admin" in token_data.roles:
            return True
        
        return any(role in token_data.roles for role in required_roles)
    
    # ============================================================================
    # 👤 User Management
    # ============================================================================
    
    def create_user(
        self,
        username: str,
        password: str,
        email: str,
        roles: List[str] = None,
        permissions: List[str] = None,
    ) -> User:
        """Create a new user"""
        
        if self._get_user_by_username(username):
            raise ValueError(f"User {username} already exists")
        
        user = User(
            id=f"user_{uuid.uuid4().hex[:8]}",
            username=username,
            email=email,
            password_hash=self.password_hasher.hash_password(password),
            roles=roles or ["user"],
            permissions=permissions or [],
        )
        
        self._users[user.id] = user
        self._users[user.username] = user
        
        logger.info(f"✅ User created: {username}")
        
        return user
    
    def update_user(self, user_id: str, **kwargs) -> Optional[User]:
        """Update user data"""
        
        user = self._get_user_by_id(user_id)
        if not user:
            return None
        
        # Update fields
        for key, value in kwargs.items():
            if key == "password":
                user.password_hash = self.password_hasher.hash_password(value)
            elif hasattr(user, key):
                setattr(user, key, value)
        
        user.updated_at = datetime.now()
        
        logger.info(f"✅ User updated: {user.username}")
        
        return user
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user"""
        
        user = self._get_user_by_id(user_id)
        if not user:
            return False
        
        del self._users[user.id]
        del self._users[user.username]
        
        logger.info(f"🗑️ User deleted: {user.username}")
        
        return True
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self._get_user_by_id(user_id)
    
    def list_users(self) -> List[User]:
        """List all users"""
        return [u for u in self._users.values() if hasattr(u, 'id')]
    
    # ============================================================================
    # 🔧 Internal Methods
    # ============================================================================
    
    def _get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self._users.get(user_id)
    
    def _get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self._users.get(username)
    
    async def _cache_user(self, user: User):
        """Cache user data in Redis"""
        
        key = f"{self._user_cache_key_prefix}{user.id}"
        await self.redis_client.set(key, user.to_dict(), ttl=3600)
    
    async def _get_cached_user(self, user_id: str) -> Optional[User]:
        """Get cached user from Redis"""
        
        key = f"{self._user_cache_key_prefix}{user_id}"
        data = await self.redis_client.get(key)
        if data:
            return User(**data)
        return None
    
    async def _blacklist_token(self, token: str, expiry: Optional[datetime] = None):
        """Add token to blacklist"""
        
        if not expiry:
            token_data = self.jwt_handler.validate_token(token)
            if token_data:
                expiry = token_data.expires_at
            else:
                expiry = datetime.now() + timedelta(hours=1)
        
        # Calculate TTL in seconds
        ttl = max(0, int((expiry - datetime.now()).total_seconds()))
        
        key = f"{self._blacklist_key_prefix}{token}"
        await self.redis_client.set(key, "blacklisted", ttl=ttl)
    
    async def _is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        
        key = f"{self._blacklist_key_prefix}{token}"
        return await self.redis_client.exists(key)


# ============================================================================
# 🔧 Dependency Functions for FastAPI
# ============================================================================

_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get auth manager singleton"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


async def authenticate_user(username: str, password: str):
    """Authenticate user and return tokens"""
    auth_manager = get_auth_manager()
    return await auth_manager.authenticate(username, password)


async def require_permission(permission: str, token_data: TokenData = None):
    """Require a specific permission"""
    if not token_data:
        raise PermissionError("Authentication required")
    
    auth_manager = get_auth_manager()
    if not auth_manager.has_permission(token_data, permission):
        raise PermissionError(f"Permission '{permission}' required")


async def require_roles(roles: List[str], token_data: TokenData = None):
    """Require one of the specified roles"""
    if not token_data:
        raise PermissionError("Authentication required")
    
    auth_manager = get_auth_manager()
    if not auth_manager.has_any_role(token_data, roles):
        raise PermissionError(f"One of roles {roles} required")


def require_auth(func):
    """Decorator for requiring authentication"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # This would be used with FastAPI dependency injection
        return await func(*args, **kwargs)
    return wrapper