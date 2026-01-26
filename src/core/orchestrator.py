import logging
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime
from core.state_store import StateStore
from ops.health import HealthMonitor
from ops.circuit_breaker import ComponentCircuitBreakers

logger = logging.getLogger("orchestrator")

class Task:
    def __init__(self, name: str, func: Callable, dependencies: List[str] = None):
        self.name = name
        self.func = func
        self.dependencies = dependencies or []
        self.status = "pending"  # pending, running, success, failed, skipped

class Orchestrator:
    """
    Unified task runner with dependency management, idempotency, and operational monitoring.
    """
    def __init__(self, state_file: str = "state/orchestrator_state.json"):
        self.state_store = StateStore(state_file)
        self.health_monitor = HealthMonitor()
        self.breakers = ComponentCircuitBreakers()
        self.tasks: Dict[str, Task] = {}

    def register_task(self, name: str, func: Callable, dependencies: List[str] = None):
        """Registers a task in the graph."""
        self.tasks[name] = Task(name, func, dependencies)

    def run_pipeline(self, pipeline_name: str, force: bool = False):
        """
        Executes registered tasks in dependency order.
        Supports state-based recovery (skips already successful tasks).
        """
        logger.info(f"Starting pipeline: {pipeline_name}")
        start_time = datetime.utcnow()
        
        # Simple topological sort/execution loop
        executed_tasks = set()
        
        while len(executed_tasks) < len(self.tasks):
            ready_tasks = [
                t for t in self.tasks.values() 
                if t.name not in executed_tasks and all(d in executed_tasks for d in t.dependencies)
            ]
            
            if not ready_tasks:
                # Potential circular dependency or missing task
                remaining = [t.name for t in self.tasks.values() if t.name not in executed_tasks]
                logger.error(f"Stuck! Cannot progress. Remaining: {remaining}")
                break
                
            for task in ready_tasks:
                self._execute_task(task, pipeline_name, force)
                executed_tasks.add(task.name)

        self.state_store.set(f"last_pipeline_{pipeline_name}", {
            "timestamp": datetime.utcnow().isoformat(),
            "duration_sec": (datetime.utcnow() - start_time).total_seconds()
        })
        self.health_monitor.record_heartbeat()
        logger.info(f"Pipeline {pipeline_name} complete.")

    def _execute_task(self, task: Task, pipeline_name: str, force: bool):
        """Executes a single task with state check and circuit breaker."""
        # Check idempotency
        checkpoint_key = f"{pipeline_name}_{task.name}"
        last_state = self.state_store.get(f"checkpoint_{checkpoint_key}")
        
        if not force and last_state and last_state.get("status") == "success":
            # For daily pipelines, we might want to check if the success was 'today'
            # For now, let's assume successful tasks stay successful unless forced
            logger.info(f"Task {task.name} already succeeded. Skipping.")
            task.status = "skipped"
            return

        logger.info(f"Executing task: {task.name}")
        task.status = "running"
        
        breaker = self.breakers.get_breaker(task.name)
        try:
            # Wrap execution with circuit breaker
            breaker.call(task.func)
            
            task.status = "success"
            self.state_store.update_checkpoint(checkpoint_key, "success")
            logger.info(f"Task {task.name} completed successfully.")
        except Exception as e:
            task.status = "failed"
            self.state_store.update_checkpoint(checkpoint_key, "failed")
            logger.error(f"Task {task.name} failed: {e}")
            raise e # Stop pipeline on failure
