from bs4 import BeautifulSoup as bs4
import requests
import json

categorical_features = []
numerical_features = []
features_file_path = 'features.txt'
value_types = set()
with open(features_file_path) as f:
    for line in f:
        feature_code, feature_name = line.split('\t')
        if feature_code == 'eid':
            continue
        url = f'https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id={feature_code}'
        response = requests.get(url)
        soup = bs4(response.text, 'html.parser')

        try:
            feature_value_type = soup.find_all(class_='txt_blu')[2].get_text()
            value_types.add(feature_value_type)
        except IndexError:
            pass
        if 'Categorical' in feature_value_type:
            categorical_features.append(feature_name)
        else:
            numerical_features.append(feature_name)

data = {
    'categorical_features': categorical_features,
    'numerical_features': numerical_features
}

with open('features_types.json', 'w') as output_file:
    json.dumps(data, indent=4)


