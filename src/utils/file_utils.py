# src/utils/file_utils.py
"""
Enterprise File Operations with Async I/O, Compression, and Encryption
"""

import os
import json
import gzip
import shutil
import hashlib
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, BinaryIO
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from cryptography.fernet import Fernet
from contextlib import asynccontextmanager

from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


class FileManager:
    """
    Enterprise File Manager with:
    - Async file operations
    - Compression support (gzip, zstd)
    - Encryption support (Fernet)
    - Directory management
    - File cleanup
    - Hash verification
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self._encryption_key = None
        
        logger.info(f"📁 FileManager initialized: {self.base_dir}")
    
    def set_encryption_key(self, key: Union[str, bytes]):
        """Set encryption key for secure file operations"""
        if isinstance(key, str):
            key = key.encode()
        self._encryption_key = key
        logger.info("🔐 Encryption key set")
    
    # ============================================================================
    # 🚀 Async File Operations
    # ============================================================================
    
    async def read_json(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Read JSON file asynchronously"""
        path = self._resolve_path(path)
        
        try:
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            logger.error(f"❌ Failed to read JSON: {path} - {str(e)}")
            raise
    
    async def write_json(
        self,
        path: Union[str, Path],
        data: Dict[str, Any],
        indent: int = 2,
        compress: bool = False,
        encrypt: bool = False,
    ):
        """Write JSON file asynchronously with optional compression and encryption"""
        path = self._resolve_path(path)
        self._ensure_directory(path.parent)
        
        content = json.dumps(data, indent=indent, default=str)
        
        # Apply compression if requested
        if compress:
            content = self._compress_data(content.encode())
        else:
            content = content.encode()
        
        # Apply encryption if requested
        if encrypt and self._encryption_key:
            content = self._encrypt_data(content)
        
        try:
            async with aiofiles.open(path, 'wb') as f:
                await f.write(content)
            logger.debug(f"✅ Written JSON: {path}")
        except Exception as e:
            logger.error(f"❌ Failed to write JSON: {path} - {str(e)}")
            raise
    
    async def read_parquet(self, path: Union[str, Path]) -> pd.DataFrame:
        """Read Parquet file asynchronously"""
        path = self._resolve_path(path)
        
        try:
            # Use pyarrow for efficient reading
            table = pq.read_table(path)
            df = table.to_pandas()
            logger.debug(f"✅ Read Parquet: {path} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to read Parquet: {path} - {str(e)}")
            raise
    
    async def write_parquet(
        self,
        path: Union[str, Path],
        df: pd.DataFrame,
        compression: str = 'snappy',
        encrypt: bool = False,
    ):
        """Write Parquet file asynchronously with optional encryption"""
        path = self._resolve_path(path)
        self._ensure_directory(path.parent)
        
        try:
            table = pa.Table.from_pandas(df)
            
            if encrypt and self._encryption_key:
                # Serialize to bytes first
                buffer = pa.BufferOutputStream()
                pq.write_table(table, buffer, compression=compression)
                data = buffer.getvalue().to_pybytes()
                encrypted = self._encrypt_data(data)
                
                async with aiofiles.open(path, 'wb') as f:
                    await f.write(encrypted)
            else:
                pq.write_table(table, str(path), compression=compression)
            
            logger.debug(f"✅ Written Parquet: {path} ({len(df)} rows)")
        except Exception as e:
            logger.error(f"❌ Failed to write Parquet: {path} - {str(e)}")
            raise
    
    # ============================================================================
    # 🔧 Utility Methods
    # ============================================================================
    
    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve path relative to base directory"""
        path = Path(path)
        if not path.is_absolute():
            path = self.base_dir / path
        return path
    
    def _ensure_directory(self, path: Path):
        """Ensure directory exists"""
        path.mkdir(parents=True, exist_ok=True)
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using gzip"""
        return gzip.compress(data, compresslevel=6)
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data"""
        return gzip.decompress(data)
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using Fernet"""
        if not self._encryption_key:
            raise ValueError("Encryption key not set")
        fernet = Fernet(self._encryption_key)
        return fernet.encrypt(data)
    
    def _decrypt_data(self, data: bytes) -> bytes:
        """Decrypt data using Fernet"""
        if not self._encryption_key:
            raise ValueError("Encryption key not set")
        fernet = Fernet(self._encryption_key)
        return fernet.decrypt(data)
    
    def get_file_hash(self, path: Union[str, Path], algorithm: str = 'sha256') -> str:
        """Get file hash"""
        path = self._resolve_path(path)
        
        hash_func = hashlib.new(algorithm)
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def safe_filename(self, filename: str) -> str:
        """Create safe filename (remove dangerous characters)"""
        import re
        return re.sub(r'[^\w\-_.]', '_', filename)
    
    def clean_old_files(
        self,
        directory: Union[str, Path],
        pattern: str = '*',
        days: int = 30,
    ) -> int:
        """Clean files older than specified days"""
        directory = self._resolve_path(directory)
        cutoff = datetime.now() - timedelta(days=days)
        cleaned = 0
        
        for file in directory.glob(pattern):
            if file.is_file() and datetime.fromtimestamp(file.stat().st_mtime) < cutoff:
                file.unlink()
                cleaned += 1
        
        logger.info(f"🧹 Cleaned {cleaned} files from {directory}")
        return cleaned
    
    def get_file_size(self, path: Union[str, Path]) -> int:
        """Get file size in bytes"""
        path = self._resolve_path(path)
        return path.stat().st_size
    
    def get_directory_size(self, directory: Union[str, Path]) -> int:
        """Get directory size in bytes"""
        directory = self._resolve_path(directory)
        total = 0
        for file in directory.rglob('*'):
            if file.is_file():
                total += file.stat().st_size
        return total


# ============================================================================
# 🔧 AsyncFileManager - Convenience Functions
# ============================================================================

class AsyncFileManager:
    """Convenience class for async file operations"""
    
    @staticmethod
    async def read_json(path: Union[str, Path]) -> Dict[str, Any]:
        """Read JSON file"""
        return await FileManager().read_json(path)
    
    @staticmethod
    async def write_json(path: Union[str, Path], data: Dict[str, Any], **kwargs):
        """Write JSON file"""
        await FileManager().write_json(path, data, **kwargs)
    
    @staticmethod
    async def read_parquet(path: Union[str, Path]) -> pd.DataFrame:
        """Read Parquet file"""
        return await FileManager().read_parquet(path)
    
    @staticmethod
    async def write_parquet(path: Union[str, Path], df: pd.DataFrame, **kwargs):
        """Write Parquet file"""
        await FileManager().write_parquet(path, df, **kwargs)


# ============================================================================
# 🔧 Convenience Functions
# ============================================================================

def compress_data(data: bytes) -> bytes:
    """Compress data using gzip"""
    return gzip.compress(data, compresslevel=6)

def decompress_data(data: bytes) -> bytes:
    """Decompress data"""
    return gzip.decompress(data)

def encrypt_file(input_path: Union[str, Path], output_path: Union[str, Path], key: bytes):
    """Encrypt a file"""
    fernet = Fernet(key)
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'rb') as f:
        data = f.read()
        encrypted = fernet.encrypt(data)
    
    with open(output_path, 'wb') as f:
        f.write(encrypted)
    
    logger.info(f"🔐 Encrypted: {input_path} -> {output_path}")

def decrypt_file(input_path: Union[str, Path], output_path: Union[str, Path], key: bytes):
    """Decrypt a file"""
    fernet = Fernet(key)
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'rb') as f:
        encrypted = f.read()
        data = fernet.decrypt(encrypted)
    
    with open(output_path, 'wb') as f:
        f.write(data)
    
    logger.info(f"🔓 Decrypted: {input_path} -> {output_path}")

async def read_json_async(path: Union[str, Path]) -> Dict[str, Any]:
    """Async JSON read"""
    return await AsyncFileManager.read_json(path)

async def write_json_async(path: Union[str, Path], data: Dict[str, Any], **kwargs):
    """Async JSON write"""
    await AsyncFileManager.write_json(path, data, **kwargs)

async def read_parquet_async(path: Union[str, Path]) -> pd.DataFrame:
    """Async Parquet read"""
    return await AsyncFileManager.read_parquet(path)

async def write_parquet_async(path: Union[str, Path], df: pd.DataFrame, **kwargs):
    """Async Parquet write"""
    await AsyncFileManager.write_parquet(path, df, **kwargs)

def get_file_hash(path: Union[str, Path], algorithm: str = 'sha256') -> str:
    """Get file hash"""
    return FileManager().get_file_hash(path, algorithm)

def safe_filename(filename: str) -> str:
    """Create safe filename"""
    import re
    return re.sub(r'[^\w\-_.]', '_', filename)

def ensure_directory(path: Union[str, Path]):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)

def clean_old_files(directory: Union[str, Path], pattern: str = '*', days: int = 30) -> int:
    """Clean old files"""
    return FileManager().clean_old_files(directory, pattern, days)


# ============================================================================
# 🔧 Singleton Instance
# ============================================================================

_file_manager: Optional[FileManager] = None


def get_file_utils() -> FileManager:
    """Get file manager singleton"""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManager(
            base_dir=config_manager.get('data.base_dir', '.')
        )
    return _file_manager