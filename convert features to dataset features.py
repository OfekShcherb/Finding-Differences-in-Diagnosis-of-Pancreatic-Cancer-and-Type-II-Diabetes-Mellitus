import json

biobank_path = "/home/ofeksh2@mta.ac.il/biobank/ukb672220.csv",
features_code_list = []
features_name_list = []
with open('C:/Users/Win10/Desktop/FinalProject/features.txt') as features_file:
    for line in features_file:
        feature_code, feature_name = line.split('\t')
        feature_name = feature_name.replace('\n', '')
        if feature_name != 'eid':
            feature_code = f'{feature_code}-0.0'
        features_code_list.append(feature_code)
        features_name_list.append(feature_name)

config = {
    'biobank_path': biobank_path,
    'features_code_list': features_code_list,
    'features_name_list': features_name_list
}

data = json.dumps(config, indent=4)

with open("C:/Users/Win10/Desktop/FinalProject/config.json", 'w') as output_file:
    output_file.write(data)

