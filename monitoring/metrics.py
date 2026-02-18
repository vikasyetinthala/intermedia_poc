from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import time
import os

# Training Metrics Registry
REGISTRY = CollectorRegistry()

# Define Metrics with Labels
TRAINING_DURATION = Gauge(
    'ml_training_duration_seconds', 
    'Time taken for the model training process', 
    labelnames=['run_id'],
    registry=REGISTRY
)

DATASET_SIZE = Gauge(
    'ml_training_dataset_size_records', 
    'Number of records used in the training window', 
    labelnames=['run_id'],
    registry=REGISTRY
)

MODEL_ACCURACY = Gauge(
    'ml_model_accuracy', 
    'Accuracy of the trained model', 
    labelnames=['run_id'],
    registry=REGISTRY
)

MODEL_F1_SCORE = Gauge(
    'ml_model_f1_score', 
    'F1 score of the trained model', 
    labelnames=['run_id'],
    registry=REGISTRY
)

MODEL_PRECISION = Gauge(
    'ml_model_precision', 
    'Precision of the trained model', 
    labelnames=['run_id'],
    registry=REGISTRY
)

MODEL_RECALL = Gauge(
    'ml_model_recall', 
    'Recall of the trained model', 
    labelnames=['run_id'],
    registry=REGISTRY
)

def push_training_metrics(metrics_dict, run_id, gateway_url="localhost:9091", job_name="ml_training_pipeline"):
    """
    Pushes collected metrics to Prometheus Pushgateway with MLflow run_id label.
    
    Args:
        metrics_dict (dict): Dictionary containing metric values.
        run_id (str): MLflow run ID to use as a label.
        gateway_url (str): Pushgateway URL.
        job_name (str): Name of the job for Prometheus tags.
    """
    try:
        # Set Gauge values with label
        if 'duration' in metrics_dict:
            TRAINING_DURATION.labels(run_id=run_id).set(metrics_dict['duration'])
        if 'dataset_size' in metrics_dict:
            DATASET_SIZE.labels(run_id=run_id).set(metrics_dict['dataset_size'])
        if 'accuracy' in metrics_dict:
            MODEL_ACCURACY.labels(run_id=run_id).set(metrics_dict['accuracy'])
        if 'f1_score' in metrics_dict:
            MODEL_F1_SCORE.labels(run_id=run_id).set(metrics_dict['f1_score'])
        if 'precision' in metrics_dict:
            MODEL_PRECISION.labels(run_id=run_id).set(metrics_dict['precision'])
        if 'recall' in metrics_dict:
            MODEL_RECALL.labels(run_id=run_id).set(metrics_dict['recall'])

        # Push to Gateway
        push_to_gateway(gateway_url, job=job_name, registry=REGISTRY)
        print(f"Successfully pushed metrics to {gateway_url} for job {job_name}")
        
    except Exception as e:
        print(f"Warning: Failed to push metrics to Pushgateway: {e}")
