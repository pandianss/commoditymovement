import os
import sys
import pandas as pd

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from governance.drift_detection import DriftDetector

class RetrainManager:
    """
    Automates model retraining based on governance triggers.
    """
    def __init__(self, drift_threshold=0.01):
        self.drift_threshold = drift_threshold
        
    def evaluate_retrain_trigger(self, drift_report_path):
        if not os.path.exists(drift_report_path):
            return False, "No drift report found."
            
        report = pd.read_csv(drift_report_path, index_col=0)
        drifted_features = report[report['drift_detected'] == True].index.tolist()
        
        if drifted_features:
            return True, f"Drift detected in features: {drifted_features}"
            
        return False, "Models are stable. No retraining required."

    def execute_retrain(self):
        """
        In a real system, this would trigger the training scripts for 
        baseline and primary models.
        """
        print("Executing retraining pipeline across all commodities...")
        # Simulated run of src/run_baseline.py and primary engines
        return "Retraining Success"

if __name__ == "__main__":
    manager = RetrainManager()
    # Check the report we just generated
    report_path = "data/processed/drift_report.csv"
    trigger, reason = manager.evaluate_retrain_trigger(report_path)
    print(f"Trigger: {trigger} | Reason: {reason}")
