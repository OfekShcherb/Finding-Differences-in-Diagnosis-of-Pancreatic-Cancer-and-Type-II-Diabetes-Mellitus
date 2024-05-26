import json
import pandas as pd


def get_people_with_disease(df, all_diseased_column, disease_code_pattern):
    people_with_disease_df = df[all_diseased_column.str.contains(disease_code_pattern)]

    return people_with_disease_df


print('loading config')
with open('/home/ofeksh2@mta.ac.il/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

biobank_path = config['biobank_path']
features_code_list = config['features_code_list']
features_name_list = config['features_name_list']

print('loading dataset')
df = pd.read_csv(biobank_path[0], usecols=(lambda x: x in features_code_list))
df_filled = df.fillna('-1')
all_diseased_column = df_filled[[f'41270-0.{i}' for i in range(100)]].agg(', '.join, axis=1)

print('splitting the dataset')
test_group_df = df.sample(n=100000)
train_group_df = df.drop(test_group_df.index)

diabetes_pattern = r'E11'
people_with_diabetes_in_test_df = get_people_with_disease(test_group_df, all_diseased_column, diabetes_pattern)
print(f'people_with_diabetes_in_test_df size: {people_with_diabetes_in_test_df.shape}')

pancreatic_cancer_pattern = r'C25'
people_with_pancreatic_cancer_in_test_df = get_people_with_disease(test_group_df, all_diseased_column,pancreatic_cancer_pattern)
print(f'people_with_pancreatic_cancer_in_test_df size: {people_with_pancreatic_cancer_in_test_df.shape}')

test_group_df.to_csv('test_data_first.csv', index=False)

people_with_diabetes_in_train_df = get_people_with_disease(train_group_df, all_diseased_column, diabetes_pattern)
print(f'people_with_diabetes_in_train_df size: {people_with_diabetes_in_train_df.shape}')

people_with_pancreatic_cancer_in_train_df = get_people_with_disease(train_group_df, all_diseased_column,pancreatic_cancer_pattern)
print(f'people_with_pancreatic_cancer_in_train_df size: {people_with_pancreatic_cancer_in_train_df.shape}')

total_number_of_patients_in_train = people_with_pancreatic_cancer_in_train_df.shape[0] + people_with_pancreatic_cancer_in_train_df.shape[0]
print(f'total_number_of_patients_in_train size: {total_number_of_patients_in_train}')

train_group_df = df.drop(people_with_diabetes_in_train_df.index)
train_group_df = df.drop(people_with_pancreatic_cancer_in_train_df.index)

train_group_df = df.sample(n=total_number_of_patients_in_train)

train_group_df.to_csv('train_data_first.csv', index=False)