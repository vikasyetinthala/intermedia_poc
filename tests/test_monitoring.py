import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Ensure the project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monitoring.metrics import push_training_metrics

class TestMetrics(unittest.TestCase):
    @patch('monitoring.metrics.push_to_gateway')
    def test_push_training_metrics(self, mock_push):
        metrics = {
            'duration': 120.5,
            'dataset_size': 1000,
            'accuracy': 0.95,
            'f1_score': 0.94,
            'precision': 0.93,
            'recall': 0.92
        }
        
        # Test push (should not raise exception)
        push_training_metrics(metrics, run_id="test_run_123", gateway_url="localhost:9091")
        
        # Verify push_to_gateway was called
        mock_push.assert_called_once()
        print("\nVerification: push_to_gateway was called correctly with mock.")

if __name__ == '__main__':
    unittest.main()
