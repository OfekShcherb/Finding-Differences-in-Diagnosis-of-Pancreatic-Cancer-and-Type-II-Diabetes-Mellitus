from Train_Model import *


start_time = time()
with open('/home/ofeksh2@mta.ac.il/config_files/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

train_path = config['train_path']

train_df = pd.read_csv(train_path, low_memory=False)

train_df = preprocessing(train_df, config)

train_cancer_healthy = train_df[train_df['Label'] != 2]
train_T2D_healthy = train_df[train_df['Label'] != 1]
train_cancer_T2D = train_df[train_df['Label'] != 0]

models = [RandomForestClassifier(), GradientBoostingClassifier(), MLPClassifier()]
models_names = ['RF', 'GB', 'MLP']

param_grids = [
        {
            'n_estimators': randint(50, 300),
            'max_depth': randint(5, 50),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 10),
            'bootstrap': [True, False],
        },
        {
            'n_estimators': randint(50, 300),
            'learning_rate': loguniform(0.0001, 1.0),
            'max_depth': randint(1, 10),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 10),
            'subsample': uniform(0.5, 0.5),
            'max_features': ['sqrt', 'log2', None],
            'loss': ['log_loss'],
        },
        {
            'hidden_layer_sizes': [(128, 64), (128, 64, 32), (128, 64, 32, 16), (256, 128, 64), (256, 128, 64, 32)],
            'activation': ['relu', 'tanh'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'max_iter': [10000],
            'alpha': loguniform(0.00001, 1.0),
        }
    ]

dfs = [train_cancer_healthy, train_T2D_healthy, train_cancer_T2D]
names = ['cancer_healthy', 'T2D_healthy', 'cancer_T2D']

for df, name in zip(dfs, names):
    print('Training ' + name)
    x_train = df.drop(['Label'], axis=1)
    y_train = df['Label']

    for model, models_name, param_grid in zip(models, models_names, param_grids):
        print(f'Fitting {models_name}...')
        grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=20, cv=3, n_jobs=-1,
                                         verbose=1, scoring='accuracy')
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_
        with open(f"{config['models_path']}/{name}_{models_name}_scaled.pkl", 'wb') as f:
            pickle.dump(best_model, f)

elapsed_time = time() - start_time
print(f'elapsed time: {timedelta(seconds=elapsed_time)}')