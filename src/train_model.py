import logging
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def train_model(
    data: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Trains a random forest classifier on specified features and target labels

    Args:
        data: The pandas dataframe containing the data for training
        config: The dictionary containing the configuration parameters

    Returns:
        The trained random forest classifier, train features, test features, train target, and test target
    """
    initial_features = config['initial_features']
    n_estimators = config['n_estimators']
    max_depth = config['max_depth']
    test_size = config['test_size']
    random_state = config['train_test_split']['random_state']

    features = data[initial_features]
    target = data[config['target_name']]

    try:
        # Split test and train data
        logging.info('Splitting data into train/test')
        train_features, test_features, train_target, test_target = train_test_split(
            features, target, test_size=test_size, random_state=random_state)

        # Train the model
        logging.info('Training the model')
        rf_classifier = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth)
        rf_classifier.fit(train_features, train_target)
    except Exception as e:
        logging.error('Failed to train model: %s', e)
        raise

    train_data = train_features.copy()
    train_data[config['target_name']] = train_target
    test_data = test_features.copy()
    test_data[config['target_name']] = test_target

    return rf_classifier, train_data, test_data


def save_data(train: pd.DataFrame, test: pd.DataFrame, save_dir: Path) -> None:
    """Saves train and test DataFrames to specified directory.

    Args:
        train: The train DataFrame to be saved
        test: The test DataFrame to be saved
        save_dir: The directory in which to save the DataFrames
    """
    try:
        logging.info('Saving train and test data')
        save_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(save_dir / 'train.csv', index=False)
        test.to_csv(save_dir / 'test.csv', index=False)
    except Exception as e:
        logging.error('Failed to save data: %s', e)
        raise


def save_model(model: RandomForestClassifier, save_path: Path) -> None:
    """Saves a trained model to a specified file.

    Args:
        model: The trained model to be saved
        save_path: The path where the trained model will be saved
    """
    try:
        logging.info('Saving model')
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        logging.info('Model saved successfully')
    except Exception as e:
        logging.error('Failed to save model: %s', e)
        raise
