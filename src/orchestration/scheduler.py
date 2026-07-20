# src/orchestration/scheduler.py
"""
Advanced Task Scheduler with Priority and Dependency Management
"""

import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import heapq
import threading

from src.logger import get_logger
from src.config_manager import config_manager

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class ScheduledTask:
    """Scheduled task definition"""
    task_id: str
    name: str
    func: Callable
    args: List = field(default_factory=list)
    kwargs: Dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    schedule: str = "once"  # once, interval, cron
    interval_seconds: int = 3600
    cron_expression: Optional[str] = None
    max_retries: int = 3
    retry_delay: int = 60
    timeout_seconds: int = 300
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "priority": self.priority.value,
            "schedule": self.schedule,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }


class TaskScheduler:
    """
    Advanced Task Scheduler with:
    - Priority-based scheduling
    - Dependency management
    - Retry logic with exponential backoff
    - Concurrent execution
    - Task timeout
    - Event-driven triggers
    - Task history and audit
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        enable_async: bool = True,
        store_history: bool = True,
    ):
        self.max_workers = max_workers
        self.enable_async = enable_async
        self.store_history = store_history
        
        # Task management
        self._tasks: Dict[str, ScheduledTask] = {}
        self._pending_queue: List[Tuple[int, str]] = []
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        
        # History
        self._history: List[ScheduledTask] = []
        self._max_history = 1000
        
        # Worker management
        self._workers: List[asyncio.Task] = []
        self._should_stop = False
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Metrics
        self._metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "cancelled_tasks": 0,
        }
        
        logger.info(f"⏰ TaskScheduler initialized")
        logger.info(f"   Max Workers: {max_workers}")
        logger.info(f"   Async: {enable_async}")
        
        # Start workers
        if enable_async:
            asyncio.create_task(self._start_workers())
    
    # ============================================================================
    # 🚀 Task Management
    # ============================================================================
    
    def schedule_task(
        self,
        name: str,
        func: Callable,
        args: List = None,
        kwargs: Dict = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        schedule: str = "once",
        interval_seconds: int = 3600,
        cron_expression: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 60,
        timeout_seconds: int = 300,
        dependencies: List[str] = None,
        tags: List[str] = None,
    ) -> str:
        """
        Schedule a new task
        
        Args:
            name: Task name
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            priority: Task priority
            schedule: Schedule type (once, interval, cron)
            interval_seconds: Interval in seconds
            cron_expression: Cron expression
            max_retries: Maximum retries on failure
            retry_delay: Retry delay in seconds
            timeout_seconds: Task timeout
            dependencies: Task dependencies
            tags: Task tags
            
        Returns:
            Task ID
        """
        
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            func=func,
            args=args or [],
            kwargs=kwargs or {},
            priority=priority,
            schedule=schedule,
            interval_seconds=interval_seconds,
            cron_expression=cron_expression,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout_seconds=timeout_seconds,
            dependencies=dependencies or [],
            tags=tags or [],
            scheduled_at=datetime.now() if schedule == "once" else None,
        )
        
        self._tasks[task_id] = task
        self._metrics["total_tasks"] += 1
        
        # Add to queue
        heapq.heappush(
            self._pending_queue,
            (task.priority.value, task_id)
        )
        
        logger.info(f"📋 Task scheduled: {name} ({task_id})")
        
        # Fire event
        asyncio.create_task(self._fire_event("task_scheduled", task))
        
        return task_id
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task"""
        
        if task_id not in self._tasks:
            return False
        
        task = self._tasks[task_id]
        
        if task.status == TaskStatus.RUNNING:
            # Cancel running task
            if task_id in self._running_tasks:
                self._running_tasks[task_id].cancel()
        
        task.status = TaskStatus.CANCELLED
        self._metrics["cancelled_tasks"] += 1
        
        logger.info(f"❌ Task cancelled: {task.name} ({task_id})")
        
        await self._fire_event("task_cancelled", task)
        
        return True
    
    async def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get task status"""
        
        if task_id not in self._tasks:
            return None
        
        return self._tasks[task_id].to_dict()
    
    async def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get task result"""
        
        if task_id not in self._tasks:
            return None
        
        return self._tasks[task_id].result
    
    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        tag: Optional[str] = None
    ) -> List[Dict]:
        """List tasks with filters"""
        
        tasks = []
        for task in self._tasks.values():
            if status and task.status != status:
                continue
            if tag and tag not in task.tags:
                continue
            tasks.append(task.to_dict())
        
        return tasks
    
    # ============================================================================
    # ⚙️ Worker Management
    # ============================================================================
    
    async def _start_workers(self):
        """Start worker tasks"""
        
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker)
        
        logger.info(f"✅ Started {self.max_workers} workers")
        
        # Wait for workers
        await asyncio.gather(*self._workers, return_exceptions=True)
    
    async def _worker_loop(self, worker_id: int):
        """Worker loop for task execution"""
        
        logger.info(f"🔧 Worker {worker_id} started")
        
        while not self._should_stop:
            try:
                # Get next task
                task_id = await self._get_next_task()
                if not task_id:
                    await asyncio.sleep(0.1)
                    continue
                
                # Execute task
                await self._execute_task(task_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Worker {worker_id} error: {str(e)}")
                await asyncio.sleep(1)
        
        logger.info(f"🔧 Worker {worker_id} stopped")
    
    async def _get_next_task(self) -> Optional[str]:
        """Get next task from queue"""
        
        async with self._lock:
            while self._pending_queue:
                priority, task_id = heapq.heappop(self._pending_queue)
                
                if task_id not in self._tasks:
                    continue
                
                task = self._tasks[task_id]
                
                # Check dependencies
                if task.dependencies:
                    deps_met = True
                    for dep_id in task.dependencies:
                        if dep_id in self._tasks:
                            dep_task = self._tasks[dep_id]
                            if dep_task.status != TaskStatus.COMPLETED:
                                deps_met = False
                                break
                    
                    if not deps_met:
                        # Requeue with same priority
                        heapq.heappush(self._pending_queue, (priority, task_id))
                        continue
                
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.SCHEDULED
                    return task_id
            
            return None
    
    async def _execute_task(self, task_id: str):
        """Execute a task"""
        
        task = self._tasks[task_id]
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        logger.info(f"▶️ Running task: {task.name} ({task_id})")
        
        try:
            # Execute with timeout
            if self.enable_async and asyncio.iscoroutinefunction(task.func):
                result = await asyncio.wait_for(
                    task.func(*task.args, **task.kwargs),
                    timeout=task.timeout_seconds
                )
            else:
                # Run sync function in thread pool
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        task.func,
                        *task.args,
                        **task.kwargs
                    ),
                    timeout=task.timeout_seconds
                )
            
            # Success
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            self._metrics["completed_tasks"] += 1
            
            logger.info(f"✅ Task completed: {task.name} ({task_id})")
            
            # Fire event
            await self._fire_event("task_completed", task)
            
        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error = f"Timeout after {task.timeout_seconds}s"
            await self._handle_task_failure(task)
            
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            self._metrics["cancelled_tasks"] += 1
            logger.info(f"⏹️ Task cancelled: {task.name} ({task_id})")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            await self._handle_task_failure(task)
        
        finally:
            # Remove from running tasks
            if task_id in self._running_tasks:
                del self._running_tasks[task_id]
            
            # Store history
            if self.store_history:
                self._history.append(task)
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]
    
    async def _handle_task_failure(self, task: ScheduledTask):
        """Handle task failure with retry logic"""
        
        self._metrics["failed_tasks"] += 1
        
        if task.retry_count < task.max_retries:
            # Retry
            task.retry_count += 1
            task.status = TaskStatus.RETRY
            
            # Exponential backoff
            backoff = task.retry_delay * (2 ** (task.retry_count - 1))
            
            logger.warning(
                f"🔄 Retrying task {task.name} ({task.task_id}) "
                f"attempt {task.retry_count}/{task.max_retries} in {backoff}s"
            )
            
            # Schedule retry
            await asyncio.sleep(min(backoff, 300))
            
            # Requeue
            async with self._lock:
                heapq.heappush(
                    self._pending_queue,
                    (task.priority.value, task.task_id)
                )
            
            await self._fire_event("task_retry", task)
            
        else:
            # Max retries exceeded
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            
            logger.error(
                f"❌ Task failed after {task.max_retries} retries: "
                f"{task.name} ({task.task_id}) - {task.error}"
            )
            
            await self._fire_event("task_failed", task)
    
    # ============================================================================
    # 🔧 Event System
    # ============================================================================
    
    def on_event(self, event_type: str, handler: Callable):
        """Register event handler"""
        self._event_handlers[event_type].append(handler)
    
    async def _fire_event(self, event_type: str, data: Any):
        """Fire an event to registered handlers"""
        
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_type, data)
                else:
                    handler(event_type, data)
            except Exception as e:
                logger.error(f"❌ Event handler error: {str(e)}")
    
    # ============================================================================
    # 🔧 Utility Methods
    # ============================================================================
    
    def trigger_task(self, name: str) -> Optional[str]:
        """Trigger a task by name (for manual execution)"""
        
        for task_id, task in self._tasks.items():
            if task.name == name and task.status == TaskStatus.PENDING:
                # Requeue
                heapq.heappush(
                    self._pending_queue,
                    (task.priority.value, task_id)
                )
                return task_id
        
        return None
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get scheduler metrics"""
        
        pending = len(self._pending_queue)
        running = len(self._running_tasks)
        
        return {
            **self._metrics,
            "pending_tasks": pending,
            "running_tasks": running,
            "total_tasks": len(self._tasks),
            "active_workers": len(self._workers),
            "history_size": len(self._history),
        }
    
    async def clear_history(self):
        """Clear task history"""
        self._history = []
        logger.info("🧹 Task history cleared")
    
    async def stop(self):
        """Stop the scheduler"""
        self._should_stop = True
        
        # Cancel running tasks
        for task_id, task in self._running_tasks.items():
            task.cancel()
        
        # Wait for workers
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        logger.info("🛑 Scheduler stopped")


# ============================================================================
# 🔧 Singleton Instance
# ============================================================================

_scheduler: Optional[TaskScheduler] = None


def get_scheduler() -> TaskScheduler:
    """Get scheduler singleton"""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler(
            max_workers=config_manager.get("scheduler.max_workers", 4),
            enable_async=True,
            store_history=True,
        )
    return _scheduler


# ============================================================================
# 🚀 Scheduler Service
# ============================================================================

class SchedulerService:
    """
    Convenience wrapper for Scheduler operations
    """
    
    def __init__(self):
        self.scheduler = get_scheduler()
    
    def schedule_retraining(
        self,
        schedule: str = "daily",
        hour: int = 2,
    ) -> str:
        """Schedule model retraining"""
        
        interval_seconds = 86400 if schedule == "daily" else 604800  # week
        
        task_id = self.scheduler.schedule_task(
            name="retrain_model",
            func=self._retrain_model,
            priority=TaskPriority.HIGH,
            schedule=schedule,
            interval_seconds=interval_seconds,
            tags=["training", "automated"],
        )
        
        logger.info(f"✅ Scheduled retraining: {task_id}")
        return task_id
    
    def schedule_drift_detection(self, interval_minutes: int = 60) -> str:
        """Schedule drift detection"""
        
        task_id = self.scheduler.schedule_task(
            name="detect_drift",
            func=self._detect_drift,
            priority=TaskPriority.MEDIUM,
            schedule="interval",
            interval_seconds=interval_minutes * 60,
            tags=["monitoring", "drift"],
        )
        
        logger.info(f"✅ Scheduled drift detection: {task_id}")
        return task_id
    
    def schedule_report_generation(self, schedule: str = "weekly") -> str:
        """Schedule report generation"""
        
        interval_seconds = 604800 if schedule == "weekly" else 86400
        
        task_id = self.scheduler.schedule_task(
            name="generate_reports",
            func=self._generate_reports,
            priority=TaskPriority.LOW,
            schedule=schedule,
            interval_seconds=interval_seconds,
            tags=["reporting"],
        )
        
        logger.info(f"✅ Scheduled report generation: {task_id}")
        return task_id
    
    async def _retrain_model(self):
        """Model retraining function"""
        from src.pipelines.training_pipeline import TrainingPipeline
        
        pipeline = TrainingPipeline()
        result = pipeline.run()
        
        return {
            "status": result["status"],
            "model_version": result.get("model_version"),
            "metrics": result.get("metrics"),
        }
    
    async def _detect_drift(self):
        """Drift detection function"""
        from src.monitoring.drift_detection import DriftDetector
        
        detector = DriftDetector()
        report = detector.detect_drift()
        
        if report.get("drift_detected", False):
            # Send alert
            await self._send_drift_alert(report)
        
        return report
    
    async def _generate_reports(self):
        """Report generation function"""
        from src.components.reporting import ClinicalReporter
        
        reporter = ClinicalReporter()
        report = reporter.generate_summary_report()
        
        return {
            "status": "success",
            "report_path": report.get("path"),
        }
    
    async def _send_drift_alert(self, report: Dict):
        """Send drift alert"""
        logger.warning(f"⚠️ Data drift detected: {report.get('drift_share')}")
        
        # Implementation for sending alerts (Slack, email, etc.)
        pass
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get scheduler metrics"""
        return await self.scheduler.get_metrics()