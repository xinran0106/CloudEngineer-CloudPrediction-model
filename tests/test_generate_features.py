import os
import yaml
import pytest
import pandas as pd
import numpy as np
import src.generate_features as gf

# load config file
def load_yaml_config(file_path: str) -> dict:
    '''
    Loads configuration file from a YAML file

    Args:
        file_path (str): The path of the YAML file

    Returns:
        dict: The loaded configuration
    '''
    with open(file_path, 'r') as yaml_file:
        local_config = yaml.safe_load(yaml_file)
    return local_config


# Build the path to the configuration file
current_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(current_dir, '..', 'config', 'default-config.yaml')
config = load_yaml_config(config_file_path)
config_features = config['generate_features']

@pytest.fixture
# Test 1: test log_entropy feature
def test_log_entropy():
    data = pd.DataFrame({
        'visible_mean': [10, 20, 30],
        'visible_max': [100, 200, 300],
        'visible_min': [1, 2, 3],
        'visible_mean_distribution': [0.5, 0.6, 0.7],
        'visible_contrast': [5, 10, 15],
        'visible_entropy': [1.2, 2.2, 3.2],
        'visible_second_angular_momentum': [0.01, 0.02, 0.03],
        'IR_mean': [50, 100, 150],
        'IR_max': [500, 1000, 1500],
        'IR_min': [10, 20, 30]
    })
    features = gf.generate_features(data, config_features)
    assert np.allclose(features['log_entropy'], np.log([1.2,2.2,3.2]))

# Test 2: unhappy test where invalid value for visible_entropy in one row of the input data
def test_log_entropy_invalid_input():
    data = pd.DataFrame({
        'visible_mean': [10, 20, 30],
        'visible_max': [100, 200, 300],
        'visible_min': [1, 2, 3],
        'visible_mean_distribution': [0.5, 0.6, 0.7],
        'visible_contrast': [5, 10, 15],
        'visible_entropy': [1.2, 2.2, 'invalid_value'],
        'visible_second_angular_momentum': [0.01, 0.02, 0.03],
        'IR_mean': [50, 100, 150],
        'IR_max': [500, 1000, 1500],
        'IR_min': [10, 20, 30]
    })
    with pytest.raises(TypeError):
        gf.generate_features(data, config_features)

# Test 3: test entropy_x_contrast feature
def test_entropy_x_contrast():
    data = pd.DataFrame({
        'visible_mean': [10, 20, 30],
        'visible_max': [100, 200, 300],
        'visible_min': [1, 2, 3],
        'visible_mean_distribution': [0.5, 0.6, 0.7],
        'visible_contrast': [5, 10, 15],
        'visible_entropy': [1.2, 2.2, 3.2],
        'visible_second_angular_momentum': [0.01, 0.02, 0.03],
        'IR_mean': [50, 100, 150],
        'IR_max': [500, 1000, 1500],
        'IR_min': [10, 20, 30]
    })
    features = gf.generate_features(data, config_features)
    assert np.allclose(features['entropy_x_contrast'], [5*1.2, 10*2.2, 15*3.2])

# Test 4: unhappy test where invalid value for visible_entropy in one row of the input data
def test_entropy_x_contrast_invalid_input():
    data = pd.DataFrame({
        'visible_mean': [10, 20, 30],
        'visible_max': [100, 200, 300],
        'visible_min': [1, 2, 3],
        'visible_mean_distribution': [0.5, 0.6, 0.7],
        'visible_contrast': [5, 10, 15],
        'visible_entropy': [1.2, 2.2, 'invalid_value'],
        'visible_second_angular_momentum': [0.01, 0.02, 0.03],
        'IR_mean': [50, 100, 150],
        'IR_max': [500, 1000, 1500],
        'IR_min': [10, 20, 30]
    })
    with pytest.raises(TypeError):
        gf.generate_features(data, config_features)

# Test 5: test IR_range feature
def test_ir_range():
    data = pd.DataFrame({
        'visible_mean': [10, 20, 30],
        'visible_max': [100, 200, 300],
        'visible_min': [1, 2, 3],
        'visible_mean_distribution': [0.5, 0.6, 0.7],
        'visible_contrast': [5, 10, 15],
        'visible_entropy': [1.2, 2.2, 3.2],
        'visible_second_angular_momentum': [0.01, 0.02, 0.03],
        'IR_mean': [50, 100, 150],
        'IR_max': [500, 1000, 1500],
        'IR_min': [10, 20, 30]
    })
    features = gf.generate_features(data, config_features)
    assert np.array_equal(features['IR_max'] - features['IR_min'], [490, 980, 1470])

# Test 6: test IR_norm_range feature
def test_ir_norm_range():
    data = pd.DataFrame({
        'visible_mean': [10, 20, 30],
        'visible_max': [100, 200, 300],
        'visible_min': [1, 2, 3],
        'visible_mean_distribution': [0.5, 0.6, 0.7],
        'visible_contrast': [5, 10, 15],
        'visible_entropy': [1.2, 2.2, 3.2],
        'visible_second_angular_momentum': [0.01, 0.02, 0.03],
        'IR_mean': [50, 100, 150],
        'IR_max': [500, 1000, 1500],
        'IR_min': [10, 20, 30]
    })
    features = gf.generate_features(data, config_features)
    assert np.array_equal(features['IR_norm_range'], [9.8, 9.8, 9.8])

# Test 7: test IR_norm_range feature with negative mean
def test_ir_norm_range_negative_mean():
    data_negative_mean = pd.DataFrame({
        'IR_mean': [-50, -100, -150],
        'IR_max': [500, 1000, 1500],
        'IR_min': [10, 20, 30],
        'visible_mean': [10, 20, 30],
        'visible_max': [100, 200, 300],
        'visible_min': [1, 2, 3],
        'visible_mean_distribution': [0.5, 0.6, 0.7],
        'visible_contrast': [5, 10, 15],
        'visible_entropy': [1.2, 2.2, 3.2],
        'visible_second_angular_momentum': [0.01, 0.02, 0.03]
    })
    features = gf.generate_features(data_negative_mean, config_features)
    assert np.array_equal(features['IR_norm_range'], [(500-10)/(-50), (1000-20)/(-100), (1500-30)/(-150)])

# Test 8: happy path test for generate_features with correct yaml parameters
def test_happy_path():
    features = config['generate_features']
    assert 'calculate_norm_range' in features
    assert 'log_transform' in features
    assert 'multiply' in features

    norm_range = features['calculate_norm_range']
    assert 'IR_norm_range' in norm_range
    assert norm_range['IR_norm_range']['min_col'] == 'IR_min'
    assert norm_range['IR_norm_range']['max_col'] == 'IR_max'
    assert norm_range['IR_norm_range']['mean_col'] == 'IR_mean'

    log_transform = features['log_transform']
    assert log_transform['log_entropy'] == 'visible_entropy'

    multiply = features['multiply']
    assert 'entropy_x_contrast' in multiply
    assert multiply['entropy_x_contrast']['col_a'] == 'visible_contrast'
    assert multiply['entropy_x_contrast']['col_b'] == 'visible_entropy'

# Test 9: unhappy path test with incorrect yaml with missing keys and invalid keys
def test_unhappy_path():
    # Test with missing keys
    missing_keys_yaml = '''
    generate_features:
      calculate_norm_range:
        IR_norm_range:
          min_col: IR_min
          max_col: IR_max
    '''
    config_load_miss = yaml.safe_load(missing_keys_yaml)
    with pytest.raises(KeyError):
        _ = config_load_miss['generate_features']['calculate_norm_range']['IR_norm_range']['mean_col']

    # Test with invalid keys
    invalid_keys_yaml = '''
    generate_features:
      calculate_norm_range:
        IR_norm_range:
          min_column: IR_min
          max_column: IR_max
          mean_column: IR_mean
    '''
    config_load_invalid = yaml.safe_load(invalid_keys_yaml)
    with pytest.raises(KeyError):
        _ = config_load_invalid['generate_features']['calculate_norm_range']['IR_norm_range']['min_col']

# Test 10: unhappy test where input DataFrame is missing required columns
def test_generate_features_missing_data():
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    config_ = {
        'log_transform': {'log_col': 'missing_col'},
        'multiply': {},
        'calculate_norm_range': {}
    }
    expected_error = KeyError
    try:
        gf.generate_features(data, config_)
    except expected_error:
        pass
    else:
        assert False
