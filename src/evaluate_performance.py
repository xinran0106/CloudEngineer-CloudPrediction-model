import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, classification_report

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def evaluate_performance(test_data: pd.DataFrame, scores: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates the performance of a binary classification model on test data

    Args:
        scores: The DataFrame containing 'ypred_proba' and 'ypred_bin' columns
        config: The dictionary containing the configuration parameters

    Returns:
        A dictionary containing:
        - auc: The area under the ROC curve
        - confusion: The confusion matrix
        - accuracy: The accuracy of the model's predictions
        - classification_report: The classification report
    """
    try:
        logging.info('Evaluating model performance')
        target_class = config['target_name']
        y_test = test_data[target_class].values
        ypred_proba_test = scores['ypred_proba']
        ypred_bin_test = scores['ypred_bin']

        metrics = {
            'auc': roc_auc_score(y_test, ypred_proba_test),
            # to list for yaml compatibility
            'confusion': confusion_matrix(y_test, ypred_bin_test).tolist(),
            'accuracy': accuracy_score(y_test, ypred_bin_test),
            'classification_report': classification_report(
                y_test, ypred_bin_test, output_dict=True  # dict for yaml compatibility
            )
        }
        logging.info('Model performance evaluated successfully')
    except Exception as e:
        logging.error('Failed to evaluate model performance: %s', e)
        raise

    return metrics


def save_metrics(metrics: Dict[str, Any], save_path: Path) -> None:
    """Saves the performance metrics to a specified file in YAML format.

    Args:
        metrics: The dictionary containing the performance metrics
        save_path: The path where the metrics will be saved
    """
    try:
        logging.info('Saving performance metrics')
        with open(save_path, 'w') as f:
            yaml.dump(metrics, f)
        logging.info('Performance metrics saved successfully')
    except Exception as e:
        logging.error('Failed to save performance metrics: %s', e)
        raise
