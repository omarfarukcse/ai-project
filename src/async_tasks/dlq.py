# src/async_tasks/dlq.py
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd

from src.logger import get_logger
from src.utils.file_utils import FileUtils

logger = get_logger(__name__)

class DeadLetterQueue:
    """
    Dead Letter Queue for failed async tasks
    Prevents silent data loss during transient failures
    """
    
    def __init__(self, storage_path: str = "data/dlq"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.queue = []
        self._load_queue()
    
    def _load_queue(self):
        """Load existing queue"""
        queue_file = self.storage_path / "dlq.json"
        if queue_file.exists():
            self.queue = FileUtils.load_json(queue_file)
            logger.info(f"Loaded {len(self.queue)} items from DLQ")
    
    def _save_queue(self):
        """Save queue to file"""
        queue_file = self.storage_path / "dlq.json"
        FileUtils.save_json(self.queue, queue_file)
    
    def enqueue(self, task_name: str, task_args: Dict, error: str,
                retry_count: int = 0, max_retries: int = 3) -> str:
        """
        Add failed task to DLQ
        
        Returns:
            dlq_id: Unique ID for the dead letter
        """
        dlq_id = f"DLQ_{int(datetime.now().timestamp())}_{len(self.queue)}"
        
        item = {
            'dlq_id': dlq_id,
            'task_name': task_name,
            'task_args': task_args,
            'error': error,
            'retry_count': retry_count,
            'max_retries': max_retries,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending',
            'processed_at': None
        }
        
        self.queue.append(item)
        self._save_queue()
        
        logger.warning(f"Added to DLQ: {dlq_id} - {task_name}")
        return dlq_id
    
    def process_dlq(self, max_items: int = 100, retry_callback=None) -> Dict[str, Any]:
        """
        Process items in DLQ with retry
        """
        results = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }
        
        pending = [item for item in self.queue if item['status'] == 'pending']
        
        for item in pending[:max_items]:
            result = self._retry_item(item, retry_callback)
            
            if result['success']:
                item['status'] = 'resolved'
                item['processed_at'] = datetime.now().isoformat()
                results['successful'] += 1
            else:
                item['retry_count'] += 1
                item['status'] = 'failed_retry' if item['retry_count'] >= item['max_retries'] else 'pending'
                results['failed'] += 1
            
            results['processed'] += 1
        
        self._save_queue()
        
        # Move to archive if still failing
        self._archive_failed()
        
        return results
    
    def _retry_item(self, item: Dict, retry_callback) -> Dict:
        """Retry a failed task"""
        try:
            if retry_callback:
                result = retry_callback(
                    item['task_name'],
                    item['task_args']
                )
                return {'success': True, 'result': result}
            else:
                # Default retry - just log
                logger.info(f"Retrying {item['dlq_id']} - {item['task_name']}")
                return {'success': False, 'error': 'No retry callback provided'}
                
        except Exception as e:
            logger.error(f"Retry failed for {item['dlq_id']}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _archive_failed(self):
        """Archive permanently failed items"""
        failed = [item for item in self.queue if item['status'] == 'failed_retry']
        
        if failed:
            archive_file = self.storage_path / f"archive_{datetime.now().strftime('%Y%m%d')}.json"
            
            if archive_file.exists():
                existing = FileUtils.load_json(archive_file)
                all_items = existing + failed
            else:
                all_items = failed
            
            FileUtils.save_json(all_items, archive_file)
            
            # Remove from queue
            self.queue = [item for item in self.queue if item['status'] != 'failed_retry']
            self._save_queue()
            
            logger.info(f"Archived {len(failed)} permanently failed items")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get DLQ statistics"""
        pending = [item for item in self.queue if item['status'] == 'pending']
        resolved = [item for item in self.queue if item['status'] == 'resolved']
        failed = [item for item in self.queue if item['status'] == 'failed_retry']
        
        return {
            'total_items': len(self.queue),
            'pending': len(pending),
            'resolved': len(resolved),
            'failed': len(failed),
            'by_task': self._get_task_distribution(),
            'age_oldest': self._get_oldest_age()
        }
    
    def _get_task_distribution(self) -> Dict[str, int]:
        """Get distribution of tasks in DLQ"""
        distribution = {}
        for item in self.queue:
            task_name = item['task_name']
            distribution[task_name] = distribution.get(task_name, 0) + 1
        return distribution
    
    def _get_oldest_age(self) -> Optional[float]:
        """Get age of oldest item in seconds"""
        if not self.queue:
            return None
        
        oldest = min(self.queue, key=lambda x: x['timestamp'])
        age = datetime.now() - datetime.fromisoformat(oldest['timestamp'])
        return age.total_seconds()