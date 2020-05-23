import os
import numpy as np
import pandas as pd
from base import BaseFeature
from google.cloud import storage
from agg_func import GroupbyTransformer, DiffGroupbyTransformer, RatioGroupbyTransformer


stats_list = ['mean', 'std', 'sum', 'max']
stats_diff_list = ['mean']
var_list = [
    "reply_engagement",
    "retweet_engagement",
    "retweet_with_comment_engagement",
    "like_engagement",
]
groupby_dict = [
    {
        'key': ['engaged_user_id'],
        'var': var_list,
        'agg': stats_list
    },
]
diff_dict = [
    {
        'key': ['engaged_user_id'],
        'var': var_list,
        'agg': stats_diff_list
    },
]


class MetaFeaturesAggByEngagedUserId(BaseFeature):
    def import_columns(self):
        return [
            "engaged_user_id"
        ]

    def make_features(self, df_train_input, df_test_input):
        df_train_features = pd.DataFrame()
        df_test_features = pd.DataFrame()

        meta_features_train = self._download_from_gs(
            feather_file_name="MetaFeatures_training.ftr"
        )
        if self.TESTING:
            meta_features_test = self._download_from_gs(
                feather_file_name="MetaFeatures_test.ftr"
            )
        else:
            meta_features_test = self._download_from_gs(
                feather_file_name="MetaFeatures_val_20200418.ftr"
            )

        target_columns = [
            "reply_engagement",
            "retweet_engagement",
            "retweet_with_comment_engagement",
            "like_engagement",
        ]

        for target_col in target_columns:
            df_train_input[target_col] = meta_features_train[f"MetaFeatures_{target_col}"].values
            df_test_input[target_col] = meta_features_test[f"MetaFeatures_{target_col}"].values

        df_total = df_train_input.append(df_test_input).reset_index(drop=True)
        org_cols = df_total.columns
        print(df_total.shape)
        print(df_total.isnull().sum())

        groupby = GroupbyTransformer(param_dict=groupby_dict)
        df_total = groupby.transform(df_total)
        diff = DiffGroupbyTransformer(param_dict=diff_dict)
        df_total = diff.transform(df_total)
        ratio = RatioGroupbyTransformer(param_dict=diff_dict)
        df_total = ratio.transform(df_total)

        new_cols = [c for c in df_total.columns if c not in org_cols]
        df_total = df_total[new_cols]
        print(f"n_features: {len(new_cols)}")

        df_train_features = df_total.iloc[:len(df_train_input)].reset_index(drop=True)
        df_test_features = df_total.iloc[len(df_train_input):].reset_index(drop=True)

        print(df_train_features.shape)
        print(df_test_features.shape)
        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    MetaFeaturesAggByEngagedUserId.main()
