import os
import numpy as np
import pandas as pd
from base import BaseFeature
from google.cloud import storage
from itertools import combinations


class MetaFeaturesComb(BaseFeature):
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

        for col1, col2 in combinations(target_columns, 2):
            new_fe_col_name = f'{col1}_{col2}'
            df_train_features[new_fe_col_name] = df_train_input[col1] + df_train_input[col2]
            df_test_features[new_fe_col_name] = df_test_input[col1] + df_test_input[col2]

        print(df_train_features.shape)
        print(df_test_features.shape)
        print(df_train_features.isnull().sum())
        print(df_test_features.isnull().sum())

        return df_train_features, df_test_features


if __name__ == "__main__":
    MetaFeaturesComb.main()
