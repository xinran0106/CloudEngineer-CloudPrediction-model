import logging
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def generate_features(features: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Generates features based on input data using specified configurations.

    Args:
        data: The input data as a Pandas DataFrame
        config: A dictionary containing the configuration

    Returns:
        A Pandas DataFrame containing the generated features
    """

    # Generate additional features based on configuration
    try:
        logging.info('Generating features')
        for key, value in config['log_transform'].items():
            features[key] = features[value].apply(np.log)
        for key, value in config['multiply'].items():
            features[key] = features[value['col_a']].multiply(features[value['col_b']])
        for key, value in config['calculate_norm_range'].items():
            features[key] = features[value['max_col']] - features[value['min_col']]
            features[key] = features[key].divide(features[value['mean_col']])
        logging.info('Feature generation completed')
    except Exception as e:
        logging.error('Failed to generate features: %s', e)
        raise

    return features


def save_dataframe(df: pd.DataFrame, save_path: Path) -> None:
    """Saves a DataFrame to a specified file.

    Args:
        df: The DataFrame to be saved
        filename: The name of the file to save the DataFrame to
    """
    try:
        logging.info('Saving DataFrame to %s', save_path)
        df.to_csv(save_path, index=False)
        logging.info('DataFrame saved successfully')
    except Exception as e:
        logging.error('Failed to save DataFrame: %s', e)
        raise
