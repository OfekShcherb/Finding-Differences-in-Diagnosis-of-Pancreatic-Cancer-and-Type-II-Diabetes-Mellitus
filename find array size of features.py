from bs4 import BeautifulSoup as bs4
import requests

features_file_path = 'C:/Users/tupe3/PycharmProjects/Finding-Differences-in-Diagnosis-of-Pancreatic-Cancer-and-Type-II-Diabetes-Mellitus/features.txt'
with open(features_file_path) as f:
    for line in f:
        feature_code, feature_name = line.split('\t')
        if feature_code == 'eid':
            continue
        url = f'https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id={feature_code}'
        response = requests.get(url)
        soup = bs4(response.text, 'html.parser')

        try:
            arrays_size = soup.find_all(class_='txt_blu')[19].get_text()
        except IndexError:
            pass
        if arrays_size != 'No':
            print(f'name: {feature_name.strip()}, code: {feature_code}, size: {arrays_size}')

# search all features with array in them, found:
#   name: Diagnoses - main ICD10, code: 41202, size: Yes (80)
#   name: Diagnoses - secondary ICD10, code: 41204, size: Yes (210)
#   name: Illnesses of father, code: 20107, size: Yes (10)
#   name: Illnesses of mother, code: 20110, size: Yes (11)
#   name: Illnesses of siblings, code: 20111, size: Yes (12)






