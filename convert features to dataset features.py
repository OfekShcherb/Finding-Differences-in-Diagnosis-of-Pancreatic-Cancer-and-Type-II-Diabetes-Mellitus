import pandas as pd
import json
from collections import defaultdict

biobank_paths = [
        "/home/ofeksh2@mta.ac.il/biobank/ukb672220.csv",
        "/home/ofeksh2@mta.ac.il/biobank/ukb673316.csv",
        "/home/ofeksh2@mta.ac.il/biobank/ukb673540.csv"
    ]
biobank_fields_paths = [
        "/home/ofeksh2@mta.ac.il/biobank/fields672220.ukb",
        "/home/ofeksh2@mta.ac.il/biobank/fields673316.ukb",
    ]

train_path = "/home/ofeksh2@mta.ac.il/datasets/train.csv"
test_path = "/home/ofeksh2@mta.ac.il/datasets/test.csv"
biobank_fields_sets = []
for path in biobank_fields_paths:
    with open(path) as fields_file:
        biobank_fields_set = set(fields_file.read().split('\n'))
        biobank_fields_sets.append(biobank_fields_set)

models_path = '/home/ofeksh2@mta.ac.il/models'

mean_imputer_path = '/home/ofeksh2@mta.ac.il/models/mean_imputer.pkl'
categorical_imputer_path = '/home/ofeksh2@mta.ac.il/models/categorical_imputer.pkl'
one_hot_encoder_path = '/home/ofeksh2@mta.ac.il/models/one_hot_encoder.pkl'
standard_scalar_path = '/home/ofeksh2@mta.ac.il/models/standard_scalar.pkl'

features_type_path = '/home/ofeksh2@mta.ac.il/config_files/features_types.json'

features_code_lists = [[], [], []]
features_name_lists = [[], [], []]
features_with_array = defaultdict(lambda: 1)
with open('/home/ofeksh2@mta.ac.il/config_files/features_with_arrays.txt') as f:
    for line in f:
        feature_code, array_size = line.split()
        features_with_array[feature_code] = int(array_size)

with open('/home/ofeksh2@mta.ac.il/config_files/features.txt') as features_file:
    for line in features_file:
        feature_code, feature_name = line.split('\t')
        feature_name = feature_name.replace('\n', '')

        size = features_with_array[feature_code]
        if feature_code in biobank_fields_sets[0]:
            list_to_add_code = features_code_lists[0]
            list_to_add_name = features_name_lists[0]
        elif feature_code in biobank_fields_sets[1]:
            list_to_add_code = features_code_lists[1]
            list_to_add_name = features_name_lists[1]
        else:
            list_to_add_code = features_code_lists[2]
            list_to_add_name = features_name_lists[2]
        for i in range(size):
            new_feature_code = f'{feature_code}-0.{i}'
            new_feature_name = f'{feature_name} - {i}'
            list_to_add_code.append(new_feature_code)
            list_to_add_name.append(new_feature_name)

config = {
    'biobank_paths': biobank_paths,
    'train_path': train_path,
    'test_path': test_path,
    'mean_imputer_path': mean_imputer_path,
    'categorical_imputer_path': categorical_imputer_path,
    'standard_scalar_path': standard_scalar_path,
    'models_path': models_path,
    'one_hot_encoder_path': one_hot_encoder_path,
    'features_code_lists': features_code_lists,
    'features_name_lists': features_name_lists,
    'features_types': features_type_path,
}

data = json.dumps(config, indent=4)

with open('/home/ofeksh2@mta.ac.il/config_files/config.json', 'w') as output_file:
    output_file.write(data)

