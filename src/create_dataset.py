import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np


def create_dataset(data_path: Path, config: Dict) -> pd.DataFrame:
    """Creates a pandas dataframe from the data in a specified file path

    Args:
        data_path: The file path of the data to be read
        config: The configuration dictionary

    Returns:
        A pandas dataframe containing the data in the specified file path
    """
    try:
        logging.info('Creating dataset')
        with open(data_path, 'r') as f:
            data = [[s for s in line.split(' ') if s!=''] for line in f.readlines()]

        # Get column names from config
        columns = config['load_data']['names']

        # Get first cloud class
        first_cloud = data[53:1077]
        first_cloud = [[float(s.replace('/n', '')) for s in cloud] for cloud in first_cloud]
        first_cloud = pd.DataFrame(first_cloud, columns=columns)
        first_cloud['class'] = np.zeros(len(first_cloud))

        # Get second cloud class
        second_cloud = data[1082:2105]
        second_cloud = [[float(s.replace('/n', '')) for s in cloud] for cloud in second_cloud]
        second_cloud = pd.DataFrame(second_cloud, columns=columns)
        second_cloud['class'] = np.ones(len(second_cloud))

        # Concatenate dataframes for training
        dataset = pd.concat([first_cloud, second_cloud])

        logging.info('Dataset created successfully')
        return dataset
    except Exception as e:
        logging.error('Failed to create dataset: %s', e)
        raise


def save_dataset(df: pd.DataFrame, save_path: Path) -> None:
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
