import json
from collections import defaultdict

biobank_path = [
        "/home/ofeksh2@mta.ac.il/biobank/ukb672220.csv",
        "/home/ofeksh2@mta.ac.il/biobank/ukb673316.csv",
        "/home/ofeksh2@mta.ac.il/biobank/ukb673540.csv"
    ]
features_code_list = []
features_name_list = []
features_with_array = defaultdict(lambda: 1)
with open('C:/Users/tupe3/PycharmProjects/Finding-Differences-in-Diagnosis-of-Pancreatic-Cancer-and-Type-II-Diabetes-Mellitus/features_with_arrays.txt') as f:
    for line in f:
        feature_code, array_size = line.split()
        features_with_array[feature_code] = int(array_size)

with open('C:/Users/tupe3/PycharmProjects/Finding-Differences-in-Diagnosis-of-Pancreatic-Cancer-and-Type-II-Diabetes-Mellitus/features.txt') as features_file:
    for line in features_file:
        feature_code, feature_name = line.split('\t')
        feature_name = feature_name.replace('\n', '')
        if feature_name == 'eid':
            continue

        size = features_with_array[feature_code]
        for i in range(size):
            new_feature_code = f'{feature_code}-0.{i}'
            new_feature_name = f'{feature_name} - {i}'
            features_code_list.append(new_feature_code)
            features_name_list.append(new_feature_name)

config = {
    'biobank_path': biobank_path,
    'features_code_list': features_code_list,
    'features_name_list': features_name_list
}

data = json.dumps(config, indent=4)

with open("C:/Users/tupe3/PycharmProjects/Finding-Differences-in-Diagnosis-of-Pancreatic-Cancer-and-Type-II-Diabetes-Mellitus/config.json", 'w') as output_file:
    output_file.write(data)

