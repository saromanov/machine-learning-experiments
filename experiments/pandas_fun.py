import pandas as pd


def basic():
    df = pd.DataFrame({'first': [0.8,0.4,0.5], 'second': [0.6,0.4,0.11]})
    result = df.ix[df.first >= 0.5, ['second']] = -1
    select = df[(df.first >= 0.5) & (df.index.isin([0,2]))]