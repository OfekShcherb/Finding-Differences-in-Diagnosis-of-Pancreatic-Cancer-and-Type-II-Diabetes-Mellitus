import json
import random

import pandas as pd
import warnings

warnings.filterwarnings("ignore")


diabetes_pattern = r'E11'
pancreatic_cancer_pattern = r'C25'
pancreatic_cancer = 1
diabetes = 2
control = 0


def get_people_with_disease(df, all_diseased_column, disease_code_pattern):
    people_with_disease_df = df[all_diseased_column.str.contains(disease_code_pattern)]

    return people_with_disease_df


def label_chunk_to_diseases(chunk):
    chunk["Label"] = chunk['Diagnoses'].apply(classify_diseases)


def classify_diseases(diseases):
    label = control
    if pancreatic_cancer_pattern in diseases:
        label = pancreatic_cancer
    elif diabetes_pattern in diseases:
        label = diabetes
    return label


print('loading config')
with open('/tmp/pycharm_project_366/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

biobank_paths = config['biobank_paths']
features_code_lists = config['features_code_lists']
features_name_list = config['features_name_list']
diagnosis_codes = [f'41270-0.{i}' for i in range(100)]
print('loading dataset')
chunk_size = 1
datasets_chunks = []
for biobank_path, features in zip(biobank_paths, features_code_lists):
    chunks = pd.read_csv(biobank_path, usecols=features, chunksize=chunk_size)
    datasets_chunks.append(chunks)

datasets_chunks = zip(*datasets_chunks)
people_in_test = people_with_disease_in_train = non_sick_in_train = counter =0
for chunk_ukb672220, chunk_ukb673316, chunk_ukb673540 in datasets_chunks:
    row_of_data = pd.concat([chunk_ukb672220, chunk_ukb673316, chunk_ukb673540], axis=1)
    row_of_data[diagnosis_codes] = row_of_data[diagnosis_codes].fillna('-1')
    all_diseased_column = row_of_data[diagnosis_codes].agg(', '.join, axis=1)
    row_of_data.drop(diagnosis_codes, axis=1)
    row_of_data['Diagnoses'] = all_diseased_column
    label_chunk_to_diseases(row_of_data)

    random_number = random.random()
    if random_number <= 1/5:
        row_of_data.to_csv(config['test_path'], mode='a', index=False)
        people_in_test = people_in_test + 1
    elif (row_of_data['Label'] > 0).all():
        row_of_data.to_csv(config['train_path'], mode='a', index=False)
        people_with_disease_in_train = people_with_disease_in_train + 1
    elif random.random() <= 1/10:
        row_of_data.to_csv(config['train_path'], mode='a', index=False)
        non_sick_in_train = non_sick_in_train + 1
    if counter % 10000 == 0:
        print(counter)

    counter += 1


print(f'Number of people in test: {people_in_test}')
print(f'Number of people with desieses in train: {people_with_disease_in_train}')
print(f'Number of people without desieses in train: {non_sick_in_train}')
print(f'Number of people in train: {people_with_disease_in_train + non_sick_in_train}')


