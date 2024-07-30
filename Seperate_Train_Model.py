import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import pickle

def classify_disease(diseases_column, disease_pattern):
    return diseases_column.str.contains(disease_pattern)

with open ('/tmp/pycharm_project_366/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

#features_code_lists = config['features_code_lists']
features_name_list = config['features_name_list']
train_path = config['train_path']
test_path = config['test_path']

train_df = pd.read_csv(train_path, low_memory=False)

features_code_dict = {}
features_with_array = defaultdict(lambda: 1)
with open('/tmp/pycharm_project_366/features_with_arrays.txt') as f:
    for line in f:
        feature_code, array_size = line.split()
        features_with_array[feature_code] = int(array_size)

with open('/tmp/pycharm_project_366/features.txt') as features_file:
    for line in features_file:
        feature_code, feature_name = line.split('\t')
        feature_name = feature_name.replace('\n', '')
        size = features_with_array[feature_code]
        for i in range(size):
            new_feature_code = f'{feature_code}-0.{i}'
            new_feature_name = f'{feature_name} - {i}'
            features_code_dict[new_feature_code] = new_feature_name

train_df = train_df.rename(columns=features_code_dict)

with open('/tmp/pycharm_project_366/features_types.json') as f:
    features_types = json.load(f)
numerical_features = [f'{feature} - 0' for feature in features_types['numerical_features']]
categorical_features = [f'{feature} - 0' for feature in features_types['categorical_features']]

mean_imputer = SimpleImputer(strategy='mean')
train_df[numerical_features] = mean_imputer.fit_transform(train_df[numerical_features])

categorical_imputer = SimpleImputer(strategy='most_frequent')
train_df[categorical_features] = categorical_imputer.fit_transform(train_df[categorical_features])

diseases_patterns = [
    ('Diabetes', r'E11'),
    ('Pancreatic Cancer', r'C25'),
    ('Obesity', r'E66'),
    ('Acute Pancreatitis', r'K85'),
    ('Alcoholic Liver Disease', r'K70'),
    ('Cirrhosis', r'K74'),
    ('Acute Hepatitis A', r'B15'),
    ('Acute Hepatitis B', r'B16'),
    ('Acute Hepatitis C', r'B171'),
    ('Toxic Liver Disease', r'K71'),
    ('Cushings Syndrome', r'E24'),
    ('Hyperthyroidism', r'E05'),
    ('Intestinal Malabsorption', r'K90'),
    ('Arterial Embolism and Thrombosis', r'I74')
]

patient_diseases = train_df['Diagnoses']
for disease, disease_pattern in diseases_patterns[2:]:
    train_df[f'Has {disease}'] = classify_disease(patient_diseases, disease_pattern)

train_df = train_df.drop(columns=['Diagnoses'])
train_df = train_df.drop(columns=[f'Diagnoses - ICD10 - {i}' for i in range(100)])

father_diagnosis_codes = [f'Illnesses of father - {i}' for i in range(10)]
mother_diagnosis_codes = [f'Illnesses of mother - {i}' for i in range(11)]
siblings_diagnosis_codes = [f'Illnesses of siblings - {i}' for i in range(12)]

father_diseases = train_df[father_diagnosis_codes].astype(str).agg(', '.join, axis=1)
mother_diseases = train_df[mother_diagnosis_codes].astype(str).agg(', '.join, axis=1)
siblings_diseases = train_df[siblings_diagnosis_codes].astype(str).agg(', '.join, axis=1)

for disease, disease_pattern in diseases_patterns:
    train_df[f'Father has {disease}'] = classify_disease(father_diseases, disease_pattern)
for disease, disease_pattern in diseases_patterns:
    train_df[f'Mother has {disease}'] = classify_disease(mother_diseases, disease_pattern)
for disease, disease_pattern in diseases_patterns:
    train_df[f'Siblings have {disease}'] = classify_disease(siblings_diseases, disease_pattern)

train_df = train_df.drop(columns=(father_diagnosis_codes + mother_diagnosis_codes + siblings_diagnosis_codes))

one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').set_output(transform='pandas')
one_hot_encoding = one_hot_encoder.fit_transform(train_df[categorical_features])
train_df = pd.concat([train_df, one_hot_encoding], axis=1)
train_df = train_df.drop(columns=categorical_features)
train_df = train_df.drop(columns=['Glycated haemoglobin (HbA1c) - 0'])

pickle.dump(one_hot_encoder, open('/tmp/pycharm_project_366/Models/One_Hot_Encoder.pk1', 'wb'))
pickle.dump(mean_imputer, open('/tmp/pycharm_project_366/Models/Mean_Imputer.pk1', 'wb'))
pickle.dump(categorical_imputer, open('/tmp/pycharm_project_366/Models/Categorical_Imputer.pk1', 'wb'))

train_cancer_healthy = train_df[train_df['Label'] != 2]
train_T2D_healthy = train_df[train_df['Label'] != 1]
train_cancer_T2D = train_df[train_df['Label'] != 0]

dfs = [train_cancer_healthy, train_T2D_healthy, train_cancer_T2D]
names = ['cancer_healthy', 'T2D_healthy', 'cancer_T2D']
for df, name in zip(dfs, names):
    print('Training ' + name)
    x_train = df.drop(['Label', 'eid'], axis=1)
    y_train = df['Label']

    gbc_model = GradientBoostingClassifier()
    gbc_model.fit(x_train, y_train)

    pickle.dump(gbc_model, open(f'/tmp/pycharm_project_366/Models/{name}.pk1', 'wb'))
