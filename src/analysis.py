import logging
from typing import List
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def save_figures(features: pd.DataFrame, config: dict, save_dir: Path) -> List[Path]:
    """Creates and saves histogram figures for each feature in a pandas dataframe

    Args:
        features: The pandas dataframe containing the features to be plotted
        config: The configuration dictionary
        save_dir: The directory in which to save the figures

    Returns:
        A list of Path objects representing the file paths of the saved figures
    """
    fig_paths = []
    target_name = config['target_name']
    target = features[target_name]

    try:
        logging.info('Creating and saving figures')
        figs = []
        for feat in features.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.hist([
                features[target == 0][feat].values, features[target == 1][feat].values
            ])
            ax.set_xlabel(' '.join(feat.split('_')).capitalize())
            ax.set_ylabel('Number of observations')
            figs.append(fig)

        # Save figures to specified directory
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, fig in enumerate(figs):
            fig_path = save_dir / ('histogram_' + str(i) + '.png')
            fig.savefig(fig_path)
            fig_paths.append(fig_path)

        logging.info('Figures saved successfully')
    except Exception as e:
        logging.error('Failed to create and save figures: %s', e)
        raise

    return fig_paths
