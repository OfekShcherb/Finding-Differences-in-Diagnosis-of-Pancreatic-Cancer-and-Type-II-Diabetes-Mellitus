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
train_path = '/tmp/pycharm_project_366/train_data.csv'
test_path = '/tmp/pycharm_project_366/test_data.csv'
biobank_fields_sets = []
for path in biobank_fields_paths:
    with open(path) as fields_file:
        biobank_fields_set = set(fields_file.read().split('\n'))
        biobank_fields_sets.append(biobank_fields_set)

features_code_lists = [[], [], []]
features_name_list = []
features_with_array = defaultdict(lambda: 1)
with open('/tmp/pycharm_project_366/features_with_arrays.txt') as f:
    for line in f:
        feature_code, array_size = line.split()
        features_with_array[feature_code] = int(array_size)

with open('/tmp/pycharm_project_366/features.txt') as features_file:
    for line in features_file:
        feature_code, feature_name = line.split('\t')
        feature_name = feature_name.replace('\n', '')
        if feature_name == 'eid':
            features_code_lists[0].append('eid')
            continue

        size = features_with_array[feature_code]
        if feature_code in biobank_fields_sets[0]:
            list_to_add = features_code_lists[0]
        elif feature_code in biobank_fields_sets[1]:
            list_to_add = features_code_lists[1]
        else:
            list_to_add = features_code_lists[2]
        for i in range(size):
            new_feature_code = f'{feature_code}-0.{i}'
            new_feature_name = f'{feature_name} - {i}'
            list_to_add.append(new_feature_code)
            features_name_list.append(new_feature_name)

config = {
    'biobank_paths': biobank_paths,
    'train_path': train_path,
    'test_path': test_path,
    'features_code_lists': features_code_lists,
    'features_name_list': features_name_list
}

data = json.dumps(config, indent=4)

with open('/tmp/pycharm_project_366/config.json', 'w') as output_file:
    output_file.write(data)

