import pandas as pd
import pickle
import json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from Train_Model import split_numerical_and_categorical_features, change_df_columns_name, fill_nans


with open('/home/ofeksh2@mta.ac.il/config_files/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

train_path = config['train_path']

train_df = pd.read_csv(train_path, low_memory=False)

train_df = change_df_columns_name(train_df, config['features_code_lists'], config['features_name_lists'])
numerical_features, categorical_features = split_numerical_and_categorical_features(config['features_types'])

#train_df = fill_nans(train_df, numerical_features)

mean_imputer = SimpleImputer(strategy='mean')
mean_imputer.fit(train_df[numerical_features])

categorical_imputer = SimpleImputer(strategy='most_frequent')
categorical_imputer.fit(train_df[categorical_features])

standard_scaler = StandardScaler()
standard_scaler.fit(train_df[numerical_features])

one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').set_output(transform='pandas')
one_hot_encoder.fit(train_df[categorical_features])

pickle.dump(one_hot_encoder, open(config['one_hot_encoder_path'], 'wb'))
pickle.dump(mean_imputer, open(config['mean_imputer_path'], 'wb'))
pickle.dump(categorical_imputer, open(config['categorical_imputer_path'], 'wb'))
pickle.dump(standard_scaler, open(config['standard_scalar_path'], 'wb'))