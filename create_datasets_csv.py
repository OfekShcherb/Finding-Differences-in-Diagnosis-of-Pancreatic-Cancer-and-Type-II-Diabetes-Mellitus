import json
import pandas as pd


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
    if diseases.contains(pancreatic_cancer_pattern):
        label = pancreatic_cancer
    elif diseases.str.contains(diabetes_pattern):
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
chunk_size = 10000
datasets_chunks = []
for biobank_path, features in zip(biobank_paths, features_code_lists):
    chunks = pd.read_csv(biobank_path, usecols=features, chunksize=chunk_size)
    datasets_chunks.append(chunks)

datasets_chunks = zip(*datasets_chunks)
for chunk_ukb672220, chunk_ukb673316, chunk_ukb673540 in datasets_chunks:
    chunk = pd.concat([chunk_ukb672220, chunk_ukb673316, chunk_ukb673540], axis=1)
    chunk[diagnosis_codes] = chunk[diagnosis_codes].fillna('-1')
    all_diseased_column = chunk[diagnosis_codes].agg(', '.join, axis=1)
    chunk.drop(diagnosis_codes, axis=1)
    chunk['Diagnoses'] = all_diseased_column
    label_chunk_to_diseases(chunk)

    test_group_df = chunk.sample(n=chunk.shape[0]//5)
    test_group_df.to_csv('test_data.csv', mode='a', index=False)
    train_group_df = chunk.drop(test_group_df.index)

    people_with_diabetes_in_train_df = get_people_with_disease(train_group_df, all_diseased_column,
                                                               diabetes_pattern)
    print(f'people_with_diabetes_in_train_df size: {people_with_diabetes_in_train_df.shape}')

    people_with_pancreatic_cancer_in_train_df = get_people_with_disease(train_group_df, all_diseased_column,
                                                                        pancreatic_cancer_pattern)
    print(f'people_with_pancreatic_cancer_in_train_df size: {people_with_pancreatic_cancer_in_train_df.shape}')

    total_number_of_patients_in_train = people_with_diabetes_in_train_df.shape[0] + \
                                        people_with_pancreatic_cancer_in_train_df.shape[0]
    print(f'total_number_of_patients_in_train size: {total_number_of_patients_in_train}')

    train_group_df = chunk.drop(people_with_diabetes_in_train_df.index)
    train_group_df = train_group_df.drop(people_with_pancreatic_cancer_in_train_df.index)
    train_group_df = train_group_df.sample(n=total_number_of_patients_in_train)
    train_group_df = pd.concat(
        [train_group_df, people_with_diabetes_in_train_df, people_with_pancreatic_cancer_in_train_df])

    train_group_df.to_csv('train_data.csv', mode='a', index=False)

print(f'finished loading dataset {biobank_path}')

