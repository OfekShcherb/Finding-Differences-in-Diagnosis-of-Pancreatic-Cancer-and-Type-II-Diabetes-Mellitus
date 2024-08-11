import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils import compute_class_weight
from sklearn.metrics import f1_score
import pickle


def split_numerical_and_categorical_features(features_types_path):
    with open(features_types_path) as f:
        features_types = json.load(f)

    numerical_features = [f'{feature} - 0' for feature in features_types['numerical_features']]
    categorical_features = [f'{feature} - 0' for feature in features_types['categorical_features']]

    return numerical_features, categorical_features


def encode_one_hot(df, one_hot_encoder_path, categorical_features):
    one_hot_encoder = pickle.load(open(one_hot_encoder_path, 'rb'))

    one_hot_encoding = one_hot_encoder.transform(df[categorical_features])
    one_hot_df = pd.concat([df, one_hot_encoding], axis=1)
    one_hot_df = one_hot_df.drop(columns=categorical_features)

    return one_hot_df


def change_df_columns_name(df, features_code_lists, features_name_lists):
    features_code_list = []
    features_code_list = [features_code_list.extend(list_to_extend) for list_to_extend in features_code_lists]

    features_name_list = []
    features_name_list = [features_name_list.extend(list_to_extend) for list_to_extend in features_name_lists]

    features_code_dict = dict(zip(features_code_list, features_name_list))
    renamed_df = df.rename(columns=features_code_dict)

    return renamed_df


def fill_nans(df, mean_imputer_path, numerical_features, categorical_imputer_path, categorical_features):
    mean_imputer = pickle.load(open(mean_imputer_path, 'rb'))
    categorical_imputer = pickle.load(open(categorical_imputer_path, 'rb'))

    filled_df = mean_imputer.transform(df[numerical_features])
    filled_df = categorical_imputer.transform(filled_df[categorical_features])

    return filled_df


def preprocessing(df, config):
    numerical_features, categorical_features = split_numerical_and_categorical_features(config['features_types'])
    df = change_df_columns_name(df, config['features_code_lists'], config['features_name_lists'])
    df = fill_nans(df, config['mean_imputer_path'], numerical_features, config['categorical_imputer_path'],
                   categorical_features)
    df = encode_one_hot(df, config['one_hot_encoder_path'], categorical_features)
    df = df.drop(columns=['Glycated haemoglobin (HbA1c) - 0'])

    return df


def classify_disease(diseases_column, disease_pattern):
    return diseases_column.str.contains(disease_pattern)


with open('/home/ofeksh2@mta.ac.il/config_files/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

train_path = config['train_path']

train_df = pd.read_csv(train_path, low_memory=False)

train_df = preprocessing(train_df, config)

y_train = train_df['Label']
x_train = train_df.drop(columns=['Label'])


models = [RandomForestClassifier(), GradientBoostingClassifier(), MLPClassifier(), KNeighborsClassifier(), SVC()]

param_grids = [
    {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
    },
    {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None],
        'loss': ['deviance', 'exponential'],
    },
    {
        'neurons_per_layer': [(128, 64), (128, 64, 32), (128, 64, 32, 16), (256, 128, 64), (256, 128, 64, 32)],
        'activation': ['relu', 'tanh'],
        'optimizer': ['adam', 'sgd', 'rmsprop'],
        'learning_rate': [0.001, 0.01, 0.1],
        'epochs': [50, 100, 200],
        'dropout_rate': [0.2, 0.3, 0.4, 0.5],
        'l2_reg': [0.0001, 0.001, 0.01],
        'batch_norm': [True, False],
    },
    {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [10, 30, 50],
    },
    {
        'C': [0.1, 1, 10, 100, 1000],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'degree': [2, 3, 4],
        'coef0': [0.0, 0.1, 0.5, 1.0],
    }
]

for i, model, param_grid in enumerate(zip(models, param_grids)):
    print(f'Fitting {i}...')
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1, scoring=f1_score())
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    with open(f"config['models_path']/{i}.pkl", 'wb') as f:
        pickle.dump(best_model, f)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
print(class_weights)
sample_weights = [class_weights[y] for y in y_train]
