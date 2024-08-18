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

def preprocessing(df, config, scaled_data=False):
    numerical_features, categorical_features = split_numerical_and_categorical_features(config['features_types'])
    df = change_df_columns_name(df, config['features_code_lists'], config['features_name_lists'])
    df = fill_nans(df, config['mean_imputer_path'], numerical_features, config['categorical_imputer_path'],
                   categorical_features)
    if scaled_data:
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
    train_df['Label'] = train_df['Label'].replace(2, 1)
    #people_with_pancreatic_cancer = train_df[train_df['label'] == 1]
    #healthy_people = train_df[train_df['label'] == 0].sample(n=people_with_pancreatic_cancer.shape[0])
    #train_df = pd.concat([healthy_people, people_with_pancreatic_cancer])

    train_df = preprocessing(train_df, config, scaled_data=True)

    y_train = train_df['Label']
    x_train = train_df.drop(columns=['Label'])

    models = [RandomForestClassifier(), GradientBoostingClassifier(), MLPClassifier()]
    models_names = ['RF', 'GB', 'MLP']

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

    #scorer = make_scorer(pancreatic_cancer_f1)
    for model, models_name, param_grid in zip(models, models_names, param_grids):
        print(f'Fitting {models_name}...')
        grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=1, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_
        with open(f"{config['models_path']}/{models_name}_scaled_healthy_vs_sick.pkl", 'wb') as f:
            pickle.dump(best_model, f)

    elapsed_time = time() - start_time
    print(f'elapsed time: {timedelta(seconds=elapsed_time)}')


