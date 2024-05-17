import pandas as pd
import json


def get_people_with_disease(df, disease_code_pattern):
    people_with_disease_df = df[df['41202-0.0'].str.contains(disease_code_pattern)]
    people_with_disease_df = pd.concat([people_with_disease_df, df[df['41204-0.0'].str.contains(disease_code_pattern)]])

    return people_with_disease_df


with open ('/home/ofeksh2@mta.ac.il/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

biobank_path = config['biobank_path']
fields = config['fields']

df = pd.read_csv(biobank_path, usecols=fields, nrows=1000)

print(df.head())
df = df.dropna()

diabetes_pattern = r'E1[0-4]*'
people_with_diabetes_df = get_people_with_disease(df, diabetes_pattern)

pancreatic_cancer_pattern = r'c25*'
people_with_pancreatic_cancer_df = get_people_with_disease(df, pancreatic_cancer_pattern)

sample_group_df = pd.concat([people_with_diabetes_df, people_with_pancreatic_cancer_df])
print(sample_group_df)



