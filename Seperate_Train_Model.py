from Train_Model import *


with open('/home/ofeksh2@mta.ac.il/config_files/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

train_path = config['train_path']

train_df = pd.read_csv(train_path, low_memory=False)

train_df = preprocessing(train_df, config)

train_cancer_healthy = train_df[train_df['Label'] != 2]
train_T2D_healthy = train_df[train_df['Label'] != 1]
train_cancer_T2D = train_df[train_df['Label'] != 0]

dfs = [train_cancer_healthy, train_T2D_healthy, train_cancer_T2D]
names = ['cancer_healthy', 'T2D_healthy', 'cancer_T2D']
for df, name in zip(dfs, names):
    print('Training ' + name)
    x_train = df.drop(['Label', 'eid'], axis=1)
    y_train = df['Label']

    gbc_model = GradientBoostingClassifier()
    gbc_model.fit(x_train, y_train)

    pickle.dump(gbc_model, open(f'/tmp/pycharm_project_366/Models/{name}.pk1', 'wb'))
