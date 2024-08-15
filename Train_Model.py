import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.utils import compute_class_weight
import pickle
from scipy.stats import randint, uniform, loguniform
from time import time
from datetime import timedelta


def pancreatic_cancer_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', labels=[1])

def split_numerical_and_categorical_features(features_types_path):
    with open(features_types_path) as f:
        features_types = json.load(f)

    numerical_features = [f'{feature} - 0' for feature in features_types['numerical_features']]
    categorical_features = [f'{feature} - 0' for feature in features_types['categorical_features']]

    return numerical_features, categorical_features

def scale_data(df, standard_scalar_path,numerical_features):
    standard_scalar = pickle.load(open(standard_scalar_path, 'rb'))

    df[numerical_features] = standard_scalar.transform(df[numerical_features])

    return df


def encode_one_hot(df, one_hot_encoder_path, categorical_features):
    one_hot_encoder = pickle.load(open(one_hot_encoder_path, 'rb'))

    one_hot_encoding = one_hot_encoder.transform(df[categorical_features])
    one_hot_df = pd.concat([df, one_hot_encoding], axis=1)
    one_hot_df = one_hot_df.drop(columns=categorical_features)

    return one_hot_df

def change_df_columns_name(df, features_code_lists, features_name_lists):
    features_code_list = []
    [features_code_list.extend(list_to_extend) for list_to_extend in features_code_lists]

    features_name_list = []
    [features_name_list.extend(list_to_extend) for list_to_extend in features_name_lists]

    features_code_dict = dict(zip(features_code_list, features_name_list))
    renamed_df = df.rename(columns=features_code_dict)

    return renamed_df

def fill_nans(df, mean_imputer_path, numerical_features, categorical_imputer_path, categorical_features):
    mean_imputer = pickle.load(open(mean_imputer_path, 'rb'))
    categorical_imputer = pickle.load(open(categorical_imputer_path, 'rb'))

    df[numerical_features] = mean_imputer.transform(df[numerical_features])
    df[categorical_features] = categorical_imputer.transform(df[categorical_features])

    return df

def preprocessing(df, config):
    numerical_features, categorical_features = split_numerical_and_categorical_features(config['features_types'])
    df = change_df_columns_name(df, config['features_code_lists'], config['features_name_lists'])
    df = fill_nans(df, config['mean_imputer_path'], numerical_features, config['categorical_imputer_path'],
                   categorical_features)
    df = scale_data(df, config['standard_scalar_path'], numerical_features)
    df = encode_one_hot(df, config['one_hot_encoder_path'], categorical_features)
    columns_to_remove = ['Glycated haemoglobin (HbA1c) - 0', 'Diagnoses']
    columns_to_remove.extend([f'Diagnoses - ICD10 - {i}' for i in range(100)])
    df = df.drop(columns=columns_to_remove)

    return df

def classify_disease(diseases_column, disease_pattern):
    return diseases_column.str.contains(disease_pattern)


if __name__ == '__main__':
    start_time = time()
    with open('/home/ofeksh2@mta.ac.il/config_files/config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    train_path = config['train_path']

    train_df = pd.read_csv(train_path, low_memory=False)

    train_df = preprocessing(train_df, config)

    y_train = train_df['Label']
    x_train = train_df.drop(columns=['Label'])


    models = [RandomForestClassifier(), GradientBoostingClassifier(), MLPClassifier()]
    models_names = ['RF', 'GB', 'MLP']
    #models_names = ['MLP']
    #models = [MLPClassifier()]

    #param_grids = [
    #    {
    #        'n_estimators': [100, 200, 300],
    #        'max_depth': [None, 10, 20, 30],
    #        'min_samples_split': [2, 5, 10],
    #        'min_samples_leaf': [1, 2, 4],
    #        'bootstrap': [True, False],
    #    },
    #    {
    #        'n_estimators': [100, 200, 300],
    #        'learning_rate': [0.01, 0.05, 0.1, 0.2],
    #        'max_depth': [3, 4, 5, 6],
    #        'min_samples_split': [2, 5, 10],
    #        'min_samples_leaf': [1, 2, 4],
    #        'subsample': [0.8, 0.9, 1.0],
    #        'max_features': ['sqrt', 'log2', None],
    #        'loss': ['log_loss', 'exponential'],
    #    },
    #    {
    #        'hidden_layer_sizes': [(128, 64), (128, 64, 32), (128, 64, 32, 16), (256, 128, 64), (256, 128, 64, 32)],
    #        'activation': ['relu', 'tanh'],
    #        'solver': ['lbfgs', 'sgd', 'adam'],
    #        'learning_rate': ['constant', 'invscaling', 'adaptive'],
    #        'max_iter': [50, 100, 200],
    #        'alpha': [0.0001, 0.001, 0.01],
    #    },
    #    {
    #        'n_neighbors': [3, 5, 7, 9, 11],
    #        'weights': ['uniform', 'distance'],
    #        'metric': ['euclidean', 'manhattan', 'minkowski'],
    #        'p': [1, 2],
    #        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    #        'leaf_size': [10, 30, 50],
    #    },
    #    {
    #        'C': [0.1, 1, 10, 100, 1000],
    #        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    #        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    #        'degree': [2, 3, 4],
    #        'coef0': [0.0, 0.1, 0.5, 1.0],
    #    }
    #]

    param_grids = [
        {
            'n_estimators': randint(50, 300),
            'max_depth': randint(5, 50),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 10),
            'bootstrap': [True, False],
        },
        {
            'n_estimators': randint(50, 300),
            'learning_rate': loguniform(0.0001, 1.0),
            'max_depth': randint(1, 10),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 10),
            'subsample': uniform(0.5, 0.5),
            'max_features': ['sqrt', 'log2', None],
            'loss': ['log_loss'],
        },
        {
            'hidden_layer_sizes': [(128, 64), (128, 64, 32), (128, 64, 32, 16), (256, 128, 64), (256, 128, 64, 32)],
            'activation': ['relu', 'tanh'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'max_iter': [10000],
            'alpha': loguniform(0.00001, 1.0),
        }
    ]
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    print(class_weights)
    sample_weights = [class_weights[y] for y in y_train]

    scorer = make_scorer(pancreatic_cancer_f1)
    for model, models_name, param_grid in zip(models, models_names, param_grids):
        print(f'Fitting {models_name}...')
        grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=20, cv=3, n_jobs=-1, verbose=1, scoring=scorer)
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_
        with open(f"{config['models_path']}/{models_name}_scaled_with_weighted_labels.pkl", 'wb') as f:
            pickle.dump(best_model, f)

    elapsed_time = time() - start_time
    print(f'elapsed time: {timedelta(seconds=elapsed_time)}')


