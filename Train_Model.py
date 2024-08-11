import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import compute_class_weight
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

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
print(class_weights)
sample_weights = [class_weights[y] for y in y_train]
mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64))
#grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
mlp_model.fit(x_train, y_train)
#grid_search.fit(x,y)
best_model = mlp_model

pickle.dump(best_model, open('/home/ofeksh2@mta.ac.il/models/Best_Model_MLP.pk1', 'wb'))
