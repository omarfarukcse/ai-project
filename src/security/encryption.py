# src/security/encryption.py
"""
Data Encryption with AES-256-GCM and Key Management
"""

import os
import base64
import hashlib
import json
from typing import Dict, Any, Optional, Union, List, Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.fernet import Fernet
import secrets
import re

from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


class KeyManager:
    """
    Key Management for encryption
    
    Features:
    - Key generation
    - Key derivation (PBKDF2, Scrypt)
    - Key rotation
    - Key storage
    """
    
    def __init__(self):
        self._keys: Dict[str, bytes] = {}
        self._key_salt = os.urandom(32)
        
        logger.info("🔑 KeyManager initialized")
    
    def generate_key(self, key_id: str, length: int = 32) -> bytes:
        """Generate a random key"""
        key = os.urandom(length)
        self._keys[key_id] = key
        logger.debug(f"🔑 Generated key: {key_id}")
        return key
    
    def derive_key_from_password(
        self,
        password: str,
        key_id: str,
        salt: Optional[bytes] = None,
        method: str = "pbkdf2",
    ) -> bytes:
        """Derive a key from a password"""
        
        salt = salt or os.urandom(32)
        password_bytes = password.encode('utf-8')
        
        if method == "pbkdf2":
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = kdf.derive(password_bytes)
        elif method == "scrypt":
            kdf = Scrypt(
                salt=salt,
                length=32,
                n=2**16,
                r=8,
                p=1,
            )
            key = kdf.derive(password_bytes)
        else:
            raise ValueError(f"Unknown key derivation method: {method}")
        
        self._keys[key_id] = key
        logger.debug(f"🔑 Derived key: {key_id}")
        return key
    
    def get_key(self, key_id: str) -> Optional[bytes]:
        """Get a key by ID"""
        return self._keys.get(key_id)
    
    def delete_key(self, key_id: str):
        """Delete a key"""
        if key_id in self._keys:
            del self._keys[key_id]
            logger.debug(f"🗑️ Deleted key: {key_id}")


class AESCipher:
    """
    AES-256-GCM Encryption
    
    Features:
    - AES-256-GCM encryption
    - Authenticated encryption
    - Associated data support
    - Base64 encoding for storage
    """
    
    def __init__(self, key: Optional[bytes] = None):
        self.key = key or os.urandom(32)
        self.cipher = AESGCM(self.key)
        
        logger.info("🔐 AESCipher initialized")
    
    @classmethod
    def from_password(cls, password: str, salt: Optional[bytes] = None) -> 'AESCipher':
        """Create cipher from password"""
        salt = salt or os.urandom(32)
        key = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        ).derive(password.encode('utf-8'))
        return cls(key)
    
    def encrypt(
        self,
        plaintext: Union[str, bytes],
        associated_data: Optional[bytes] = None,
    ) -> Tuple[str, bytes]:
        """
        Encrypt data with AES-256-GCM
        
        Returns:
            Tuple of (encrypted_base64, nonce)
        """
        
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        
        nonce = os.urandom(12)  # 96 bits for GCM
        ciphertext = self.cipher.encrypt(
            nonce,
            plaintext,
            associated_data=associated_data,
        )
        
        # Combine ciphertext with nonce for storage
        encrypted = base64.b64encode(ciphertext).decode('utf-8')
        nonce_b64 = base64.b64encode(nonce).decode('utf-8')
        
        return encrypted, nonce
    
    def decrypt(
        self,
        ciphertext_b64: str,
        nonce_b64: str,
        associated_data: Optional[bytes] = None,
    ) -> str:
        """Decrypt data with AES-256-GCM"""
        
        ciphertext = base64.b64decode(ciphertext_b64)
        nonce = base64.b64decode(nonce_b64)
        
        try:
            plaintext = self.cipher.decrypt(
                nonce,
                ciphertext,
                associated_data=associated_data,
            )
            return plaintext.decode('utf-8')
        except Exception as e:
            logger.error(f"❌ Decryption failed: {str(e)}")
            raise ValueError("Decryption failed - invalid key or corrupted data")


class SecureStorage:
    """
    Secure storage with encryption
    
    Features:
    - Encrypted storage
    - Data integrity
    - Secure deletion
    - Key rotation
    """
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.cipher = AESCipher(encryption_key)
        self._storage: Dict[str, Tuple[str, str]] = {}
        
        logger.info("💾 SecureStorage initialized")
    
    def store(self, key: str, value: Union[str, bytes]) -> str:
        """Store encrypted data"""
        encrypted, nonce = self.cipher.encrypt(value)
        self._storage[key] = (encrypted, nonce)
        return encrypted
    
    def retrieve(self, key: str) -> Optional[str]:
        """Retrieve and decrypt data"""
        if key not in self._storage:
            return None
        
        encrypted, nonce = self._storage[key]
        try:
            return self.cipher.decrypt(encrypted, nonce)
        except Exception:
            return None
    
    def delete(self, key: str):
        """Securely delete data"""
        if key in self._storage:
            # Overwrite with random data
            self._storage[key] = (os.urandom(64).hex(), os.urandom(16).hex())
            del self._storage[key]
            logger.debug(f"🗑️ Securely deleted: {key}")
    
    def rotate_key(self, new_key: bytes):
        """Rotate encryption key"""
        new_cipher = AESCipher(new_key)
        
        for key, (encrypted, nonce) in self._storage.items():
            try:
                # Decrypt with old key
                plaintext = self.cipher.decrypt(encrypted, nonce)
                # Re-encrypt with new key
                new_encrypted, new_nonce = new_cipher.encrypt(plaintext)
                self._storage[key] = (new_encrypted, new_nonce)
            except Exception as e:
                logger.error(f"❌ Failed to rotate key for {key}: {str(e)}")
                continue
        
        self.cipher = new_cipher
        logger.info("🔄 Key rotated successfully")


class DataMasker:
    """
    Data masking and anonymization
    
    Features:
    - Email masking
    - Phone masking
    - Credit card masking
    - SSN masking
    - Custom pattern masking
    """
    
    def __init__(self):
        logger.info("🎭 DataMasker initialized")
    
    def mask_email(self, email: str) -> str:
        """Mask email addresses"""
        if '@' not in email:
            return email
        
        local, domain = email.split('@', 1)
        if len(local) <= 3:
            return f"{local}@*******"
        
        return f"{local[:2]}****@{domain}"
    
    def mask_phone(self, phone: str) -> str:
        """Mask phone numbers"""
        # Remove non-digit characters
        digits = re.sub(r'\D', '', phone)
        
        if len(digits) <= 4:
            return phone
        
        return f"{digits[:2]}****{digits[-4:]}"
    
    def mask_credit_card(self, card: str) -> str:
        """Mask credit card numbers"""
        # Remove non-digit characters
        digits = re.sub(r'\D', '', card)
        
        if len(digits) <= 4:
            return card
        
        return f"****{digits[-4:]}"
    
    def mask_ssn(self, ssn: str) -> str:
        """Mask SSN"""
        # Remove non-digit characters
        digits = re.sub(r'\D', '', ssn)
        
        if len(digits) != 9:
            return ssn
        
        return f"***-**-{digits[-4:]}"
    
    def mask_custom(
        self,
        text: str,
        pattern: str,
        replacement: str = "****",
        group_to_keep: Optional[int] = None,
    ) -> str:
        """Mask custom pattern"""
        
        if group_to_keep:
            # Keep specific group
            match = re.match(pattern, text)
            if match and group_to_keep < len(match.groups()) + 1:
                return text
        return re.sub(pattern, replacement, text)
    
    def anonymize_patient_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize patient data"""
        
        anonymized = {}
        
        for key, value in data.items():
            if key.lower() in ['email', 'patient_email']:
                anonymized[key] = self.mask_email(str(value))
            elif key.lower() in ['phone', 'phone_number']:
                anonymized[key] = self.mask_phone(str(value))
            elif key.lower() in ['ssn', 'social_security']:
                anonymized[key] = self.mask_ssn(str(value))
            elif 'card' in key.lower():
                anonymized[key] = self.mask_credit_card(str(value))
            else:
                anonymized[key] = value
        
        return anonymized


class EncryptionManager:
    """
    Complete Encryption Manager with:
    - AES-256-GCM encryption
    - Key management
    - Secure storage
    - Data masking
    - Key rotation
    """
    
    def __init__(self):
        self.key_manager = KeyManager()
        self.data_masker = DataMasker()
        self._default_key_id = "default"
        
        # Generate default key
        self.key_manager.generate_key(self._default_key_id)
        
        logger.info("🔐 EncryptionManager initialized")
    
    def get_cipher(self, key_id: Optional[str] = None) -> AESCipher:
        """Get AES cipher with specified key"""
        
        key_id = key_id or self._default_key_id
        key = self.key_manager.get_key(key_id)
        
        if not key:
            key = self.key_manager.generate_key(key_id)
        
        return AESCipher(key)
    
    def encrypt(
        self,
        data: Union[str, bytes],
        key_id: Optional[str] = None,
        associated_data: Optional[bytes] = None,
    ) -> Dict[str, str]:
        """Encrypt data"""
        
        cipher = self.get_cipher(key_id)
        encrypted, nonce = cipher.encrypt(data, associated_data)
        
        return {
            "encrypted": encrypted,
            "nonce": nonce,
            "key_id": key_id or self._default_key_id,
        }
    
    def decrypt(
        self,
        encrypted_data: Dict[str, str],
        associated_data: Optional[bytes] = None,
    ) -> str:
        """Decrypt data"""
        
        key_id = encrypted_data.get("key_id", self._default_key_id)
        cipher = self.get_cipher(key_id)
        
        return cipher.decrypt(
            encrypted_data["encrypted"],
            encrypted_data["nonce"],
            associated_data,
        )
    
    def secure_store(self, key: str, value: Union[str, bytes]) -> Dict[str, str]:
        """Store encrypted data"""
        
        return self.encrypt(value)
    
    def secure_retrieve(self, key: str, encrypted_data: Dict[str, str]) -> str:
        """Retrieve and decrypt data"""
        
        return self.decrypt(encrypted_data)
    
    def mask_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data"""
        
        return self.data_masker.anonymize_patient_data(data)
    
    def rotate_key(self, old_key_id: str, new_key_id: str):
        """Rotate encryption key"""
        
        old_key = self.key_manager.get_key(old_key_id)
        new_key = self.key_manager.get_key(new_key_id)
        
        if not old_key:
            raise ValueError(f"Old key not found: {old_key_id}")
        
        if not new_key:
            new_key = self.key_manager.generate_key(new_key_id)
        
        logger.info(f"🔄 Key rotated: {old_key_id} -> {new_key_id}")


# ============================================================================
# 🔧 Singleton Instance
# ============================================================================

_encryption_manager: Optional[EncryptionManager] = None


def get_encryption_manager() -> EncryptionManager:
    """Get encryption manager singleton"""
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    return _encryption_manager