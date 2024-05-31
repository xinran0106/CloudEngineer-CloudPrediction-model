import logging
from typing import Dict, Any
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def score_model(test_data: pd.DataFrame, model: RandomForestClassifier, config: Dict[str, Any]) -> pd.DataFrame:
    """Scores the model on the test data

    Args:
        test_data: The test data as a Pandas DataFrame
        model: The trained model to be scored
        config: The dictionary containing the configuration parameters

    Returns:
        A Pandas DataFrame containing the predicted probabilities and binary predictions for the test data
    """
    initial_features = config['initial_features']

    try:
        logging.info('Scoring the model')
        ypred_proba_test = model.predict_proba(test_data[initial_features])[:, 1]
        ypred_bin_test = model.predict(test_data[initial_features])

        results = pd.DataFrame(
            {'ypred_proba': ypred_proba_test, 'ypred_bin': ypred_bin_test})
        logging.info('Model scored successfully')
    except Exception as e:
        logging.error('Failed to score model: %s', e)
        raise

    return results

def save_scores(scores: pd.DataFrame, save_path: Path) -> None:
    """Saves the scores to a specified file.

    Args:
        scores: The DataFrame to be saved
        save_path: The path where the scores will be saved
    """
    try:
        logging.info('Saving scores')
        scores.to_csv(save_path, index=False)
        logging.info('Scores saved successfully')
    except Exception as e:
        logging.error('Failed to save scores: %s', e)
        raise
