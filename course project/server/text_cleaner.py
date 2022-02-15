from sklearn.base import BaseEstimator, TransformerMixin
import re


def clean_text(text):
    delimiters = ' ', '\n', '_'
    regex_pattern = '|'.join(map(re.escape, delimiters))
    return ' '.join(re.split(regex_pattern, re.sub(r'[0-9]+', '', text.lower())))


def clean_series(s):
    for i in range(s.shape[0]):
        s.iloc[i] = clean_text(s.iloc[i].lower())


class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        clean_series(X)
        return X
