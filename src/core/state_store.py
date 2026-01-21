import json
import os
from typing import Any, Dict, Optional
from datetime import datetime

class StateStore:
    """
    Manages durable state for the system to ensure idempotency and recovery.
    """
    def __init__(self, state_file_path: str):
        self.path = state_file_path
        self._state = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self._state, f, indent=4)

    def set(self, key: str, value: Any):
        self._state[key] = value
        self._state["last_updated"] = datetime.utcnow().isoformat()
        self.save()

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def update_checkpoint(self, job_name: str, status: str, marker: Optional[str] = None):
        checkpoint = {
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "marker": marker
        }
        self.set(f"checkpoint_{job_name}", checkpoint)

    def get_all(self) -> Dict[str, Any]:
        return self._state
