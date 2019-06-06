import numpy as np
import pandas as pd
import pickle
from collections import Counter

from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('../sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    
    df.fillna(0, inplace=True)

    return tickers,df, hm_days

def buy_sell_hold(*args):
    cols = [c for c in args]
    req = 0.02

    for col in cols:
        if col > req:
            return 0.025
        elif col < -req:
            return -0.025
        else:
            return 0

def extract_featuresets(ticker):
    tickers, df, hm_days = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, *[df['{}_{}d'.format(ticker, i)]for i in range(1, hm_days+1)]))

    vals = df['{}_target'.format(ticker)].values
    str_vals = [str(i) for i in vals]

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    return X, y, df

extract_featuresets("XOM")


def run_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.25)

    # classifier = neighbors.KNeighborsClassifier()
    classifier = VotingClassifier([('lsvc', svm.LinearSVC()),
                                   ('knn', neighbors.KNeighborsClassifier()),
                                   ('rfor', RandomForestClassifier())])

    classifier.fit(X_train,y_train)
    confidence = classifier.score(X_test, y_test)
    print('Accuracy: ', confidence)
    predictions = classifier.predict(X_test)

    print("Predicted Spread: ", Counter(predictions))

    return confidence

run_ml("BAC")