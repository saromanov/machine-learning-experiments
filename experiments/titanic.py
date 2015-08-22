import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from scipy.stats import mode
from sklearn.grid_search import GridSearchCV

def load_test():
    df = pd.read_csv('../pyconuk-introtutorial/data/test.csv')
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    df['Gender']= df['Sex'].map({'female':0, 'male': 1}).astype(int)
    age_mean = df['Age'].mean()
    df['Age'] = df['Age'].fillna(age_mean)
    fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')


def model1():
    df = pd.read_csv('../pyconuk-introtutorial/data/train.csv')
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    df['Gender']= df['Sex'].map({'female':0, 'male': 1}).astype(int)
    age_mean = df['Age'].mean()
    mode_embarked = mode(df['Embarked'])[0][0]
    df['Embarked'] = df['Embarked'].fillna(mode_embarked)
    df['Age'] = df['Age'].fillna(age_mean)
    df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)
    df = df.drop(['Sex', 'Embarked'], axis=1)
    cols = df.columns.tolist()
    cols = [cols[1]] + cols[0:1] + cols[2:]
    df = df[cols]
    train_data = df.values
    model = Pipeline([
        ('inp', preprocessing.Imputer(strategy='mean', missing_values=-1)),
        ('clf', RandomForestClassifier(n_estimators=100)),
        ('cls2', svm.SVC()),
        ])
    grid = GridSearchCV(model, {
        'inp__strategy': ['mean', 'median'],
        'clf__max_features': [0.5,1],
        'clf__max_depth': [5, None],
        }, cv=5, verbose=3)
    grid.fit(train_data[0:, 2:], train_data[0:,0])
    model = RandomForestClassifier(n_estimators=100)
    model = model.fit(train_data[0:, 2:], train_data[0:,0])

    df_test = pd.read_csv('../pyconuk-introtutorial/data/test.csv')
    df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    df_test['Gender']= df_test['Sex'].map({'female':0, 'male': 1}).astype(int)
    age_mean = df_test['Age'].mean()
    df_test['Age'] = df_test['Age'].fillna(age_mean)
    fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')
    df_test['Fare'] = df_test[['Fare', 'Pclass']].apply(lambda x: fare_means[x['Pclass']]
        if pd.isnull(x['Fare']) else x['Fare'], axis=1)
    df_test = pd.concat([df_test, pd.get_dummies(df_test['Embarked'], prefix='Embarked')], axis=1)
    df_test = df_test.drop(['Sex', 'Embarked'], axis=1)
    test_data = df_test.values
    output = model.predict(test_data[:,1:])
    print("This is test: ", model.score(train_data[0:, 2:], train_data[0:,0]))

model1()

