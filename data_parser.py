"""
A data parser to get data ready for models to train.
"""
import pandas as pd

class DataParser(object):
    def __init__(self, df_train=None, df_test=None, numeric_vars=[], category_vars=[], ignore_vars=[]):
        assert ((df_train is not None) and (df_test is not None))
        assert not ((len(numeric_vars) == 0) and (len(category_vars) == 0))
        self.df_train = df_train
        self.df_test = df_test
        self.numeric_vars = numeric_vars
        self.category_vars = category_vars
        self.ignore_vars = ignore_vars
        self.feature_index = None
        self.feature_size = None
        self.field_size = None
        self.gen_feature_index()

    def gen_feature_index(self):
        df = pd.concat([self.df_train, self.df_test], sort=False)
        feature_index = {}
        fi, field_size = 0, 0
        for col in df.columns:
            if col in self.ignore_vars:
                continue
            if col in self.numeric_vars:
                feature_index[col] = fi
                fi += 1
                field_size += 1
            elif col in self.category_vars:
                value_unique, value_cnt = df[col].unique(), df[col].nunique()
                feature_index[col] = dict(zip(value_unique, range(fi, fi + value_cnt)))
                fi += value_cnt
                field_size += 1
        self.feature_index = feature_index
        self.feature_size = fi
        self.field_size = field_size

    def parse(self, data, label_col='LABEL'):
        data_v, data_fi = data.copy(), data.copy()
        y = data[label_col].to_list()
        for col in data.columns:
            if col in self.numeric_vars:
                data_fi[col] = self.feature_index[col]
            elif col in self.category_vars:
                data_fi[col] = data_fi[col].map(self.feature_index[col])
                data_v[col] = 1.0
            else:
                data_fi.drop(col, axis=1, inplace=True)
                data_v.drop(col, axis=1, inplace=True)
        Xv = data_v.values
        Xi = data_fi.values
        return Xv, Xi, y
