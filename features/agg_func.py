import itertools
import numpy as np
import pandas as pd

"""https://github.com/okotaku/pet_finder/blob/master/code/all_tools.py
"""

class GroupbyTransformer():
    def __init__(self, param_dict=None):
        self.param_dict = param_dict

    def _get_params(self, p_dict):
        key = p_dict['key']
        if 'var' in p_dict.keys():
            var = p_dict['var']
        else:
            var = self.var
        if 'agg' in p_dict.keys():
            agg = p_dict['agg']
        else:
            agg = self.agg
        if 'on' in p_dict.keys():
            on = p_dict['on']
        else:
            on = key
        return key, var, agg, on

    def _aggregate(self, dataframe):
        self.features = []
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            all_features = list(set(key + var))
            new_features = self._get_feature_names(key, var, agg)
            features = dataframe[all_features].groupby(key)[
                var].agg(agg).reset_index()
            features.columns = key + new_features
            self.features.append(features)
        return self

    def _merge(self, dataframe, merge=True):
        for param_dict, features in zip(self.param_dict, self.features):
            key, var, agg, on = self._get_params(param_dict)
            if merge:
                dataframe = dataframe.merge(features, how='left', on=on)
            else:
                new_features = self._get_feature_names(key, var, agg)
                dataframe = pd.concat([dataframe, features[new_features]], axis=1)
        return dataframe

    def transform(self, dataframe):
        self._aggregate(dataframe)
        return self._merge(dataframe, merge=True)

    def _get_feature_names(self, key, var, agg):
        _agg = []
        for a in agg:
            if not isinstance(a, str):
                _agg.append(a.__name__)
            else:
                _agg.append(a)
        return ['_'.join([a, v, 'groupby'] + key) for v in var for a in _agg]

    def get_feature_names(self):
        self.feature_names = []
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            self.feature_names += self._get_feature_names(key, var, agg)
        return self.feature_names

    def get_numerical_features(self):
        return self.get_feature_names()


class DiffGroupbyTransformer(GroupbyTransformer):
    def _aggregate(self):
        raise NotImplementedError

    def _merge(self):
        raise NotImplementedError

    def transform(self, dataframe):
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            for a in agg:
                for v in var:
                    new_feature = '_'.join(['diff', a, v, 'groupby'] + key)
                    base_feature = '_'.join([a, v, 'groupby'] + key)
                    dataframe[new_feature] = dataframe[base_feature] - dataframe[v]
        return dataframe

    def _get_feature_names(self, key, var, agg):
        _agg = []
        for a in agg:
            if not isinstance(a, str):
                _agg.append(a.__name__)
            else:
                _agg.append(a)
        return ['_'.join(['diff', a, v, 'groupby'] + key) for v in var for a in _agg]


class RatioGroupbyTransformer(GroupbyTransformer):
    def _aggregate(self):
        raise NotImplementedError

    def _merge(self):
        raise NotImplementedError

    def transform(self, dataframe):
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            for a in agg:
                for v in var:
                    new_feature = '_'.join(['ratio', a, v, 'groupby'] + key)
                    base_feature = '_'.join([a, v, 'groupby'] + key)
                    dataframe[new_feature] = dataframe[v] / dataframe[base_feature]
        return dataframe

    def _get_feature_names(self, key, var, agg):
        _agg = []
        for a in agg:
            if not isinstance(a, str):
                _agg.append(a.__name__)
            else:
                _agg.append(a)
        return ['_'.join(['ratio', a, v, 'groupby'] + key) for v in var for a in _agg]
