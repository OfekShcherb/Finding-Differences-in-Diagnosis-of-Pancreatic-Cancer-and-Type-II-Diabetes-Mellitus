import pandas as pd

biobank_path = '/home/ofeksh2@mta.ac.il/biobank/ukb672220.csv'
fields = ['eid', '2966']

df = pd.read_csv(biobank_path, usecols=fields, nrows=100)

print(df)



