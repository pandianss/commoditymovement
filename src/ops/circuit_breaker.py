from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """
    Implements circuit breaker pattern for failure isolation.
    """
    def __init__(self, 
                 failure_threshold: int = 3,
                 timeout_seconds: int = 60,
                 success_threshold: int = 2):
        self.failure_threshold = failure_threshold
        self.timeout = timedelta(seconds=timeout_seconds)
        self.success_threshold = success_threshold
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Executes function through circuit breaker.
        """
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if datetime.utcnow() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise Exception(f"Circuit breaker OPEN for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """
        Handles successful call.
        """
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """
        Handles failed call.
        """
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def reset(self):
        """
        Manually resets circuit breaker.
        """
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0

class ComponentCircuitBreakers:
    """
    Manages circuit breakers for multiple system components.
    """
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(self, component_name: str) -> CircuitBreaker:
        """
        Gets or creates a circuit breaker for a component.
        """
        if component_name not in self.breakers:
            self.breakers[component_name] = CircuitBreaker()
        return self.breakers[component_name]
    
    def get_status(self) -> Dict[str, str]:
        """
        Returns status of all circuit breakers.
        """
        return {name: breaker.state.value for name, breaker in self.breakers.items()}
