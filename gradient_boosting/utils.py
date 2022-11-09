import pandas as pd
import numpy as np



def get_categorical_features(df):
    """Identify categorical features in feature dataset."""
    cat = [i  for i,(col, dt) in enumerate(df.dtypes.items()) if dt == object]
    return cat


def clean_currency(s):
    """Converts currency string '$1,234' to float 1234.0 """
    s = ''.join([x for x in s if x not in ['$',',']])
    s = float(s)
    return s


def clean_data(df):
    '''Removes NaN and converts string cash values to numeric'''
    for i, col in enumerate(df):
        series = df[col]
        sample = series.iloc[0]
        if isinstance(sample, str) and sample[0] == '$':
            series = series.apply(lambda x : clean_currency(x) if pd.notnull(x) else x)

        if i in get_categorical_features(df):
            series = series.apply(lambda x: str(x))
            series = series.fillna(value = 'nan')
        else: 
            series = series.fillna(value = np.nan)

        df[col] = series

    return df


def load_data(path, test = False): 
    df = pd.read_csv(path)
    df = df.sample(frac=1).reset_index(drop=True)
    if test:
        y = None
        X = df.drop(['INDEX', 'TARGET_AMT', 'TARGET_FLAG'], axis = 1)
        X = clean_data(X)
    else: 
        df = df.drop(['INDEX', 'TARGET_AMT'], axis = 1)
        df = clean_data(df)
        y = df['TARGET_FLAG']
        X = df.drop('TARGET_FLAG', axis = 1)

    return X, y


