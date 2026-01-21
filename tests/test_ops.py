import unittest
import os
import json
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core.state_store import StateStore
from ops.health import HealthMonitor
from ops.drift import DriftDetector
from ops.circuit_breaker import CircuitBreaker, CircuitState

class TestOperationalIntegrity(unittest.TestCase):
    
    def test_state_store_persistence(self):
        """Test that StateStore persists and recovers state."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            # Create and save state
            store1 = StateStore(temp_path)
            store1.set("test_key", "test_value")
            store1.update_checkpoint("job1", "success", "marker123")
            
            # Load in new instance
            store2 = StateStore(temp_path)
            self.assertEqual(store2.get("test_key"), "test_value")
            checkpoint = store2.get("checkpoint_job1")
            self.assertEqual(checkpoint["status"], "success")
            self.assertEqual(checkpoint["marker"], "marker123")
        finally:
            os.unlink(temp_path)
    
    def test_health_monitor_freshness(self):
        """Test data freshness checking."""
        monitor = HealthMonitor()
        
        # Fresh data
        recent = datetime.utcnow() - timedelta(minutes=30)
        result = monitor.check_data_freshness(recent, max_age_hours=2)
        self.assertEqual(result["status"], "healthy")
        
        # Stale data
        old = datetime.utcnow() - timedelta(hours=5)
        result = monitor.check_data_freshness(old, max_age_hours=2)
        self.assertEqual(result["status"], "stale")
    
    def test_drift_detector_psi(self):
        """Test PSI calculation for drift detection."""
        # Create baseline distribution
        np.random.seed(42)
        baseline = pd.Series(np.random.normal(0, 1, 1000))
        
        detector = DriftDetector(baseline)
        
        # Similar distribution - no drift
        current_similar = pd.Series(np.random.normal(0, 1, 1000))
        is_drifting, psi = detector.check_drift(current_similar, threshold=0.2)
        self.assertFalse(is_drifting)
        self.assertLess(psi, 0.2)
        
        # Different distribution - drift
        current_shifted = pd.Series(np.random.normal(2, 1, 1000))  # Mean shifted
        is_drifting, psi = detector.check_drift(current_shifted, threshold=0.2)
        self.assertTrue(is_drifting)
        self.assertGreater(psi, 0.2)
    
    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions."""
        breaker = CircuitBreaker(failure_threshold=2, timeout_seconds=1)
        
        # Initially closed
        self.assertEqual(breaker.state, CircuitState.CLOSED)
        
        # Simulate failures
        def failing_func():
            raise Exception("Test failure")
        
        # First failure
        with self.assertRaises(Exception):
            breaker.call(failing_func)
        self.assertEqual(breaker.state, CircuitState.CLOSED)
        
        # Second failure - should open
        with self.assertRaises(Exception):
            breaker.call(failing_func)
        self.assertEqual(breaker.state, CircuitState.OPEN)
        
        # Subsequent calls should fail immediately
        with self.assertRaises(Exception):
            breaker.call(failing_func)
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        import time
        breaker = CircuitBreaker(failure_threshold=1, timeout_seconds=1, success_threshold=1)
        
        # Trigger open
        with self.assertRaises(Exception):
            breaker.call(lambda: 1/0)
        
        self.assertEqual(breaker.state, CircuitState.OPEN)
        
        # Wait for timeout
        time.sleep(1.1)
        
        # Should transition to half-open and allow test
        def success_func():
            return "success"
        
        result = breaker.call(success_func)
        self.assertEqual(result, "success")
        self.assertEqual(breaker.state, CircuitState.CLOSED)

if __name__ == '__main__':
    unittest.main()
