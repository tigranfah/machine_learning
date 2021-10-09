import numpy as np
import pandas as pd

df = pd.read_csv('yerevan_train.csv', index_col=0)

def get_district_value(district):
    """Get value basied on district.
    
    parameters:
    district (str): district name.
    
    returns:
    int: corresponsing value for given district.
    """
    return {
        'Center': 13,
        'Arabkir': 4,
        # 'Shengavit': 3,
        # 'Avan': 3,
        # 'Malatia-Sebastia': 3,
        # 'Qanaqer-Zeytu': 3,
        # 'Nor Norq': 3,
        # 'Davtashen': 3,
        # 'Erebuni': 3,
        # 'Norq Marash': 0,
        # 'Nubarashen': 0,
        # 'Vahagni district': 0
    }.get(district, 0)

def get_condition_value(condition):
    """Get value basied on condition.
    
    parameters:
    condition (str): apartment condition.
    
    returns:
    int: corresponsing value for given apartment condition.
    """
    return {
        'newly repaired': 4,
        'good': 2,
        # 'zero condition': 1
    }.get(condition, 0)

def get_building_type_value(building_type):
    """Get value basied on building_type.
    
    parameters:
    building_type (str): material used in apartment.
    
    returns:
    int: corresponsing value for given building type.
    """
    return {
        'monolit': 5,
        'other': 3,
        'stone': 3,
        'panel': 2
    }.get(building_type, 0)

"""Create 'streets' dict for hloding all unique street names and 
   their value based on the amount of house prices there."""
"""I have used a global variable here for simplicity, because I only cared for decreasing the lost fucntion."""
streets = {}
for num, value in enumerate(df.street.unique()):
    index = np.where(df.street.unique() == value)[0][0]
    length = df.street.unique().shape[0]
    streets[value] = 10 if index <= length//3 else 5
    # streets[value] = 10 if index <= length//3 else 5 if length//3 < index <= length//6 else 2

def get_feature_value(feature_name, feature):
    """Access all created values for features.
    
    parameters:
    feature_name (str): name of apartment feature.
    feature (str): apartment feature itself.
    feature (int/float): already measured value for given feature.
    
    returns:
    (int): corresponging value for given feature name and feature.
    """
    if not type(feature) is str:
        return {
            'floor': 5 if feature <= 16 else 0,
            'max_floor': 4 if feature <= 16 else 1
        }.get(feature_name, feature)
    return {
        'building_type': get_building_type_value(feature),
        'district': get_district_value(feature),
        'condition': get_condition_value(feature),
        'street': streets.get(feature, 0)
    }.get(feature_name, feature)


def featurize(apartment):
    """
    :param datum: single house information in dict
    :return: (x, y) tuple, where x is a numpy vector, and y is a number
    """

    """Add single apartment features(except thous that are removed) to 'features' list."""
    features = [1]
    price = apartment.pop('price')
    del apartment['region']
    del apartment['url']
    
    features.append(apartment.pop('area'))
    for feature_name in apartment:
        features.append(get_feature_value(feature_name, apartment[feature_name]))
        
    return np.array(features), price

def fit_ridge_regression(X, Y, l=0.1):
    """
    :param X: A numpy matrix, where each row is a data element (X)
    :param Y: A numpy vector of responses for each of the rows (y)
    :param l: ridge variable
    :return: A vector containing the hyperplane equation (beta)
    """
    # TODO, fill ridge regression part
    X, Y = np.array(X), np.array(Y)
    beta = np.linalg.inv(l * np.eye(X.shape[1]) + X.T.dot(X)).dot(X.T).dot(Y)
    return beta


def get_data(df):
    """
    :param df: all apartment information in DataFrame
    :return: X, Y, where X(each row is a data element) is matrix and Y(response vector)
    """
    X, Y = [], []
    for _, datum in df.iterrows():
        x, y = featurize(dict(datum))
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)
